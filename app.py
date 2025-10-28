#
# fork from https://huggingface.co/spaces/OmniAICreator/Anime-Llasa-3B-Captions-Demo
# modified for local environment instead of huggingface spaces
# import 44.1kHz fixed code from https://files.catbox.moe/6lm1wv.py
# 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from xcodec2.configuration_bigcodec import BigCodecConfig
from xcodec2.modeling_xcodec2 import XCodec2Model
import torchaudio
import gradio as gr
import re
import argparse

parser = argparse.ArgumentParser(description='Anime-Llasa-3B-Captions-Demo ローカル版')
parser.add_argument('--full-cpu', action='store_true')
args = parser.parse_args()
if args.full_cpu:
    print("fully running on cpu")

# -------------------------------
# Model IDs (adjust if needed)
# -------------------------------
llasa_model_id = 'NandemoGHS/Anime-Llasa-3B-Captions'
xcodec2_model_id = "NandemoGHS/Anime-XCodec2-44.1kHz"

# -------------------------------
# Lazy/global objects (kept close to original)
# NOTE: Do NOT .cuda() at import-time to keep ZeroGPU-compatible.
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(llasa_model_id)
model = AutoModelForCausalLM.from_pretrained(
    llasa_model_id,
    torch_dtype=torch.bfloat16,
)
model.eval()  # device move happens inside infer()

ckpt_path = hf_hub_download(repo_id=xcodec2_model_id, filename="model.safetensors")
ckpt = {}
with safe_open(ckpt_path, framework="pt", device="cpu") as f:
    for k in f.keys():
        ckpt[k.replace(".beta", ".bias")] = f.get_tensor(k)
    codec_config = BigCodecConfig.from_pretrained(xcodec2_model_id)
    codec_model = XCodec2Model.from_pretrained(
        None, config=codec_config, state_dict=ckpt
    )
    codec_model.eval()

whisper_turbo_pipe = None  # created on-demand only if needed (inside infer)

# -------------------------------
# Normalization (aligned with preprocessing)
# -------------------------------
REPLACE_MAP: dict[str, str] = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",
    r"　": "",
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",  # dashes
    r"[\uff5e\u301C]": "ー",  # wave dash variants
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}

FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
        )
    }
)
_HALFWIDTH_KATAKANA_CHARS = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
_FULLWIDTH_KATAKANA_CHARS = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    _HALFWIDTH_KATAKANA_CHARS, _FULLWIDTH_KATAKANA_CHARS
)
FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {chr(full): chr(half) for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))}
)

INVALID_PATTERN = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    r"\u0041-\u005A\u0061-\u007A"
    r"\u0030-\u0039"
    r"。、「」、!?…♪♡○（）]"  # allow （）
)

def normalize(text: str) -> str:
    """Normalize text to match the preprocessing rules."""
    for pattern, replacement in REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = text.translate(FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(HALFWIDTH_KATAKANA_TO_FULLWIDTH)
    text = re.sub(r"…{3,}", "……", text)
    return text

# -------------------------------
# Utilities for codec token strings
# -------------------------------
def ids_to_speech_tokens(speech_ids):
    """Convert int ids like 12345 to '<|s_12345|>' strings."""
    return [f"<|s_{int(sid)}|>" for sid in speech_ids]

def extract_speech_ids(token_strs):
    """
    Parse a list of token strings (e.g., ['<|s_12|>', '<|s_34|>', ...])
    into a list of ints [12, 34, ...]. Ignore non-speech tokens safely.
    """
    out = []
    for tok in token_strs:
        if tok.startswith('<|s_') and tok.endswith('|>'):
            try:
                out.append(int(tok[4:-2]))
            except ValueError:
                continue
    return out

def build_system_text(meta: dict) -> str:
    """
    Build system text exactly like preprocessing (fixed order/keys).
    """
    def v(key: str) -> str:
        val = meta.get(key)
        return val if val else ""
    return (
        f"emotion: {v('emotion')}\n"
        f"profile: {v('profile')}\n"
        f"mood: {v('mood')}\n"
        f"speed: {v('speed')}\n"
        f"prosody: {v('prosody')}\n"
        f"pitch_timbre: {v('pitch_timbre')}\n"
        f"style: {v('style')}\n"
        f"notes: {v('notes')}\n"
        f"caption: {v('caption')}"
    )

# -------------------------------
# Main inference
# -------------------------------
def infer(
    sample_audio_path,
    ref_text,
    target_text,
    caption,       # required
    emotion,
    profile,
    mood,
    speed,
    prosody,
    pitch_timbre,
    style,
    notes,
    temperature,
    top_p,
    repetition_penalty,
    progress=gr.Progress()
):
    # Basic checks
    if not caption or not caption.strip():
        gr.Error("Caption is required.")
        return None
    if not target_text or not target_text.strip():
        gr.Warning("Please input text to generate audio.")
        return None
    if len(target_text) > 300:
        gr.Warning("Text is too long. Trimming to 300 characters.")
        target_text = target_text[:300]
    target_text = normalize(target_text)

    # Device setup (ZeroGPU-safe)
    device = torch.device('cuda' if not args.full_cpu and torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    codec_model.to(device).eval()

    # Use codec sampling rate for resampling/outputs
    sr_input = getattr(codec_model.config, 'input_sampling_rate', 16000)
    sr_output = getattr(codec_model.config, 'sampling_rate', 44100)

    with torch.no_grad():
        speech_prefix_token_strs = []
        prefix_token_ids = []
        prompt_wav_len = 0
        input_text = target_text

        # Optional reference audio for style prompting
        if sample_audio_path:
            progress(0, 'Loading and trimming audio...')
            waveform, sample_rate = torchaudio.load(sample_audio_path)

            # Trim to 15 seconds if too long
            if waveform.size(-1) / sample_rate > 15:
                gr.Warning("Trimming reference audio to first 15 seconds.")
                waveform = waveform[..., : sample_rate * 15]

            # Ensure mono
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to codec SR
            if sample_rate != sr_input:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr_input)(waveform)

            prompt_wav = waveform  # (1, T) at sr_codec
            prompt_wav_len = prompt_wav.shape[1]

            # Reference text: provided or transcribed
            if ref_text and ref_text.strip():
                prompt_text = normalize(ref_text.strip())
                progress(0.5, 'Using provided reference text. Encoding audio...')
            else:
                progress(0.25, 'Transcribing reference audio with Whisper...')
                if not args.full_cpu:
                    # llasaモデルをCPUに移動してVRAMを解放 VRAMのキャッシュをクリア
                    print("[VRAM_SWAP] Llasa model moved to CPU.")
                    model.to("cpu")
                    torch.cuda.empty_cache()
                global whisper_turbo_pipe
                if whisper_turbo_pipe is None:
                    whisper_turbo_pipe = pipeline(
                        "automatic-speech-recognition",
                        model="openai/whisper-large-v3-turbo",
                        torch_dtype=torch.float16,
                        device=device.type if not args.full_cpu else "cpu",
                    )
                else:
                    if not args.full_cpu:
                        print("[Whisper] Moving Whisper model to CUDA (FP16) for transcription.")
                        whisper_turbo_pipe.model.to("cuda").half()
                prompt_text = whisper_turbo_pipe(prompt_wav[0].cpu().numpy())['text'].strip()
                if not args.full_cpu:
                    # whisperモデルをCPUに移動してVRAMを解放 llasaモデルをCUDAに戻してVRAMのキャッシュをクリア
                    print("[VRAM_SWAP] Whisper model moved to CPU. Llasa model moved to CUDA")
                    whisper_turbo_pipe.model.to("cpu").half()
                    model.to("cuda")
                    torch.cuda.empty_cache()
                progress(0.5, 'Transcribed! Encoding reference audio...')
            print("REFRENECE TEXT: " + prompt_text)

            # Encode prompt audio to codec tokens
            vq_code_prompt = codec_model.encode_code(input_waveform=prompt_wav.to(device))[0, 0, :].tolist()
            speech_prefix_token_strs = ids_to_speech_tokens(vq_code_prompt)
            prefix_token_ids = tokenizer.convert_tokens_to_ids(speech_prefix_token_strs)

            # Concatenate ref text + target text (same behavior as before)
            input_text = (prompt_text + ' ' + target_text).strip()

        progress(0.75, "Building prompt & generating audio...")

        # Build system text from metadata (caption required)
        meta = {
            "emotion": emotion or "",
            "profile": profile or "",
            "mood": mood or "",
            "speed": speed or "",
            "prosody": prosody or "",
            "pitch_timbre": pitch_timbre or "",
            "style": style or "",
            "notes": notes or "",
            "caption": caption.strip(),
        }
        system_text = build_system_text(meta)

        # Chat template: system -> user -> assistant (continue assistant)
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
        assistant_content = "<|SPEECH_GENERATION_START|>" + ''.join(speech_prefix_token_strs)

        chat = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": assistant_content}
        ]

        input_ids = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True
        ).to(device)

        text = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            continue_final_message=True
        )
        print(text)

        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        # Generate
        outputs = model.generate(
            input_ids,
            max_length=2048,  # match training
            eos_token_id=speech_end_id,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        # Extract generated speech tokens (skip prompt part)
        if sample_audio_path and len(prefix_token_ids) > 0:
            start_idx = input_ids.shape[1] - len(prefix_token_ids)
        else:
            start_idx = input_ids.shape[1]
        generated_ids = outputs[0][start_idx:-1]  # drop <|SPEECH_GENERATION_END|>

        speech_token_strs = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
        speech_ids = extract_speech_ids(speech_token_strs)

        if not speech_ids:
            gr.Error("Audio generation produced no speech tokens.")
            return None

        # Decode codec tokens to waveform
        code_tensor = torch.tensor(speech_ids, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
        gen_wav = codec_model.decode_code(code_tensor)  # [1, 1, T] at sr_codec

        # If prefix used, strip prefix-duration to return only generated part
        if sample_audio_path and prompt_wav_len > 0:
            scaled_len = int(prompt_wav_len * sr_output / sr_input)
            gen_wav = gen_wav[:, :, scaled_len:]

        progress(1, 'Synthesized!')
        return (sr_output, gen_wav[0, 0, :].detach().cpu().numpy())

# ===============================
# UI: TTS Tab (unchanged behavior + metadata fields)
# ===============================
with gr.Blocks() as app_tts:
    gr.Markdown("# Anime Llasa 3B (with metadata-aware system prompt)")

    with gr.Row():
        with gr.Column():
            ref_audio_input = gr.Audio(label="Reference Audio (optional)", type="filepath")
            ref_text_input = gr.Textbox(
                label="Reference Text (Optional)",
                placeholder="If you provide reference audio, you can optionally provide its transcript here. If left empty, it will be transcribed automatically.",
                lines=3
            )
            gen_text_input = gr.Textbox(label="Text to Generate", lines=8)

        with gr.Column():
            gr.Markdown("### System Metadata (caption is required)")
            caption_input = gr.Textbox(label="caption (REQUIRED)", placeholder="音声を説明する短いキャプション", lines=2)
            emotion_input = gr.Textbox(label="emotion", placeholder="感情（happy / sad / seriousなど）")
            profile_input = gr.Textbox(label="profile", placeholder="話者プロファイル（お姉さん的な女性声、若い男性声、大人の女性声など）")
            mood_input = gr.Textbox(label="mood", placeholder="ムード（恥ずかしさ、悲しみ、愛情的など）")
            speed_input = gr.Textbox(label="speed", placeholder="話速（ゆっくり、速い、一定など）")
            prosody_input = gr.Textbox(label="prosody", placeholder="抑揚・リズム（震え声、平坦、語尾が上がるなど）")
            pitch_timbre_input = gr.Textbox(label="pitch_timbre", placeholder="ピッチ・声質（高め、中低音、息多め、囁きなど）")
            style_input = gr.Textbox(label="style", placeholder="スタイル（ナレーション風、会話帳、囁き、喘ぎなど）")
            notes_input = gr.Textbox(label="notes", placeholder="特記事項（距離感、吐息などの追加事項）", lines=2)

    with gr.Row():
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.05, label="Temperature")
        top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Top-p")
        repetition_penalty_slider = gr.Slider(minimum=1.0, maximum=1.5, value=1.1, step=0.05, label="Repetition Penalty")

    generate_btn = gr.Button("Synthesize", variant="primary")
    audio_output = gr.Audio(label="Synthesized Audio")

    generate_btn.click(
        infer,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            caption_input,
            emotion_input,
            profile_input,
            mood_input,
            speed_input,
            prosody_input,
            pitch_timbre_input,
            style_input,
            notes_input,
            temperature_slider,
            top_p_slider,
            repetition_penalty_slider,
        ],
        outputs=[audio_output],
    )

# ===============================
# UI: Examples Tab
# ===============================

# Preset examples (None/None（空） -> "")
EXAMPLES = {
    "ex0": {
        "text": "今思えば、あの時すでに運命の歯車は狂っていたのだろう",
        "caption": "落ち着いた中低音の女性声。シリアスな雰囲気で、張りのある声で断定的に話す。ナレーションのようなスタイル。",
        "emotion": "serious",
        "profile": "落ち着いた女性声",
        "mood": "シリアス、深刻",
        "speed": "一定",
        "prosody": "メリハリがある、断定的",
        "pitch_timbre": "中低音、張りのある声",
        "style": "ナレーション風",
        "notes": "",
    },
    "ex1": {
        "text": "ちょっと、何触ってるの！？痴漢はれっきとした犯罪よ！この変態！",
        "caption": "大人の女性の声。冷静に問い詰め始め、次第に語気を強めていく。最後には怒りを込めて張りのある声で言い放つ。",
        "emotion": "angry",
        "profile": "大人の女性声",
        "mood": "詰問、怒り",
        "speed": "普通",
        "prosody": "メリハリが強く、最後は語気が強まる",
        "pitch_timbre": "中低音、張りのある声",
        "style": "会話調",
        "notes": "事実を突きつけるような強い口調。",
    },
    "ex2": {
        "text": "あ、あのね……ずっと言えなかったけど…。私、ずっとあなたのことが好きでした。つ、付き合ってください！",
        "caption": "恥ずかしそうに話す若い女性の声。泣き出しそうな震え声で、途切れ途切れに想いを伝える。切なさがこもっている。",
        "emotion": "shy",
        "profile": "若い女性声",
        "mood": "恥ずかしさ、切なさ",
        "speed": "遅い",
        "prosody": "途切れがち、感情がこもっている",
        "pitch_timbre": "震え声、高め、息多め",
        "style": "告白",
        "notes": "泣き出しそうな震え声。",
    },
    "ex3": {
        "text": "（腹を押しつぶされて）（うめき声）うっ…！（すすり泣き）やめて…もう殴らないでぇ…",
        "caption": "幼い少女がお腹を殴られ苦しそうにうめく。高くてか細いロリ声で、泣き出しそうな震え声。",
        "emotion": "sad",
        "profile": "ロリ声",
        "mood": "怯え、悲しみ",
        "speed": "",
        "prosody": "震え声",
        "pitch_timbre": "高め、か細い",
        "style": "嗚咽、うめき声",
        "notes": "泣き出しそうなか細いうめき声。",
    },
    "ex4": {
        "text": "（すすり泣き）…っ…うぅ…ぁぁ…",
        "caption": "若い女性が言葉にならず、感情を押し殺してすすり泣いている。悲しみや悔しさがこもった嗚咽。",
        "emotion": "sad",
        "profile": "若い女性声",
        "mood": "悲しみ、悔しさ",
        "speed": "",
        "prosody": "",
        "pitch_timbre": "鼻にかかった声",
        "style": "嗚咽、すすり泣き",
        "notes": "言葉にならず、感情を押し殺してすすり泣いている様子。",
    },
    "ex5": {
        "text": "そういえばあの時、庭で彼女を見かけたような…。",
        "caption": "落ち着いた低めの若い男性声。思い出すように、淡々とした口調で話す。一定のペースと抑揚。",
        "emotion": "worried",
        "profile": "若い男性声",
        "mood": "落ち着き",
        "speed": "",
        "prosody": "落ち着いている",
        "pitch_timbre": "やや低め",
        "style": "会話調、独り言",
        "notes": "思い出すように話している。",
    },
    "ex6": {
        "text": "（耳舐め）はむっ、れろっ…ちゅっ…（含み笑い）ふふっ…（耳元で）どう？（囁き）お耳、気持ちいい？",
        "caption": "若い女性のセクシーな耳舐めと囁き声。吐息を多く含んだ低めのトーンで、ゆっくりと誘うように問いかける。耳元で話しているような距離感。",
        "emotion": "seductive",
        "profile": "成熟した女性声",
        "mood": "誘惑的、セクシー",
        "speed": "とても遅い",
        "prosody": "語尾が上がる",
        "pitch_timbre": "低め、息多め、囁き",
        "style": "耳舐め、囁き",
        "notes": "耳舐めをしながら、非常に近い距離感で話す。",
    },
    "ex7": {
        "text": "（喘ぎ）ん、はぁんっ…あっ、んんぅ！",
        "caption": "若い女性の喘ぎ声。セリフはなく、苦悶と快感が混じったうめき声が続く。高めの声で、絶頂に至るような様子。",
        "emotion": "ecstatic",
        "profile": "若い女性声",
        "mood": "快楽、絶頂",
        "speed": "",
        "prosody": "",
        "pitch_timbre": "高めの喘ぎ声",
        "style": "喘ぎ、うめき声",
        "notes": "セリフはなく、苦悶と快感が混じった喘ぎ声のみ。",
    },
    "ex8": {
        "text": "（喘ぎ）はあんっ、あっ、あふっ、ふああっ、ふあっ、あっ、ああんっ！そんなに激しく動いたら…あんっ、イク、イっちゃう！",
        "caption": "若い女性のリズミカルな喘ぎ声。高い声で息を漏らしながら、連続して喘いでいる。快感が続いている様子。",
        "emotion": "ecstatic",
        "profile": "若い女性声",
        "mood": "快楽、絶頂",
        "speed": "",
        "prosody": "リズミカルな喘ぎ",
        "pitch_timbre": "高め、息多め",
        "style": "喘ぎ",
        "notes": "連続した喘ぎ声。ピストン運動を想起させるようなリズミカルな息遣い。",
    },
    "ex9": {
        "text": "（囁き）ほぉら、もっといっぱい突いてぇ…（喘ぎ）んっ、あんっ…あっ、んん…。良いわよ、君のおちんちん、気持ちいい…（喘ぎ）あんっ",
        "caption": "お姉さんのような低めの囁きと喘ぎ声。大人びた声で、余裕ありげに快楽の声を上げる。",
        "emotion": "aroused",
        "profile": "お姉さん的な女性声",
        "mood": "快楽",
        "speed": "ゆっくり",
        "prosody": "吐息混じり",
        "pitch_timbre": "低め",
        "style": "喘ぎ",
        "notes": "甘い囁きと喘ぎ声。",
    },
    "ex10": {
        "text": "（フェラ音）あむっ、ちゅっ、（チュパ音）ちゅぱっ、ちゅぷっ……。（含み笑い）ふふっ、すごくビクビクしてる",
        "caption": "若い女性の声。愛情を込めたフェラチオを思わせるウェットなチュパ音が続く。官能的な雰囲気。",
        "emotion": "aroused",
        "profile": "若い女性声",
        "mood": "官能的、愛情",
        "speed": "とても遅い",
        "prosody": "語尾が上がる",
        "pitch_timbre": "息多め、ウェットな音質",
        "style": "キス音、チュパ音",
        "notes": "キスやフェラチオを想起させるチュパ音が続く。ウェットなリップノイズが特徴的。",
    },
}

def _none_to_blank(x: str | None) -> str:
    """Map various 'None' markers to empty string."""
    if x is None:
        return ""
    x_str = str(x).strip()
    if x_str.lower() == "none" or x_str == "None（空）" or x_str == "None(空)":
        return ""
    return x_str

def apply_example(example_key: str):
    """Return values in the TTS input order to auto-fill the fields."""
    ex = EXAMPLES.get(example_key, {})
    return (
        ex.get("text", ""),
        ex.get("caption", ""),
        _none_to_blank(ex.get("emotion")),
        _none_to_blank(ex.get("profile")),
        _none_to_blank(ex.get("mood")),
        _none_to_blank(ex.get("speed")),
        _none_to_blank(ex.get("prosody")),
        _none_to_blank(ex.get("pitch_timbre")),
        _none_to_blank(ex.get("style")),
        _none_to_blank(ex.get("notes")),
    )

with gr.Blocks() as app_examples:
    gr.Markdown("## Examples\nChoose a preset and apply it to the TTS tab inputs.")

    # Build readable labels for the dropdown
    labels = [f"{k}: {EXAMPLES[k]['caption'][:60]}…" if len(EXAMPLES[k]['caption']) > 60 else f"{k}: {EXAMPLES[k]['caption']}"
              for k in EXAMPLES.keys()]
    keys = list(EXAMPLES.keys())
    label_to_key = {labels[i]: keys[i] for i in range(len(keys))}

    example_dropdown = gr.Dropdown(choices=labels, value=labels[0], label="Select an example")
    apply_btn = gr.Button("Apply to TTS", variant="primary")

    # When clicked, write to the TTS inputs
    apply_btn.click(
        lambda label: apply_example(label_to_key[label]),
        inputs=[example_dropdown],
        outputs=[
            # Map to TTS components in order: gen_text -> caption -> emotion -> ...
            # (We do not touch ref audio/text here)
            gen_text_input,
            caption_input,
            emotion_input,
            profile_input,
            mood_input,
            speed_input,
            prosody_input,
            pitch_timbre_input,
            style_input,
            notes_input,
        ],
    )

# ===============================
# UI: README Tab (Japanese)
# ===============================
README_MD = r"""
# README（推論用メタデータの書き方）

このデモは、systemメタデータ（`emotion / profile / mood / speed / prosody / pitch_timbre / style / notes / caption`）を
systemメッセージとして厳密な行順で渡すことで、TTS の読み方・声色を制御します。caption は必須です。

---

## 各フィールドの意味と記述例

- **caption（必須）**  
  後段の TTS 制御に使う**短い日本語キャプション**。**セリフ本文は書かない**。  
  - 推奨：**1〜2文、全角 30〜80 文字**。  
  - 例：「落ち着いた中低音の女性声。シリアスで張りがあり、断定的に語るナレーション調。」

- **emotion**（1つ選択／英語タグ）  
  例：`angry, sad, excited, surprised, ecstatic, shy, aroused, serious, relaxed, joyful, ...`  
  学習時のリストと整合する英語タグを 1 つ。

- **profile**（話者プロファイル）  
  例：「お姉さん的な女性声」「若い男性声」「落ち着いた男性声」「大人の女性声」

- **mood**（感情・ムードの自然文）  
  例：「シリアス」「快楽」「恥ずかしさ」「落ち着き」「官能的」

- **speed**（話速／自由記述）  
  例：「とても遅い」「やや速い」「一定」「(1.2×)」

- **prosody**（抑揚・リズム）  
  例：「メリハリがある」「語尾が上がる」「ため息混じり」「平坦」「震え声」

- **pitch_timbre**（ピッチ／声質）  
  例：「高め」「低め」「中低音」「息多め」「張りのある」「囁き」「鼻にかかった声」

- **style**（発話スタイル）  
  例：「ナレーション風」「会話調」「朗読調」「囁き」「喘ぎ」「嗚咽」「告白」

- **notes**（特記事項）  
  間・ブレス・笑い、効果音の有無、距離感（耳元／遠くから）など。必要なければ空でOK。

---

## emotionの一覧

以下のようなemotionを利用可能です。ただしemotionは自動アノテーションなので、ものによっては出現していない・学習データ量が極端に少ないものもあると思われます。その場合の効果は薄いです。

"angry", "sad", "disdainful", "excited", "surprised", "satisfied", "unhappy", "anxious", "hysterical", "delighted", "scared", "worried", "indifferent", "upset", "impatient", "nervous", "guilty", "scornful", "frustrated", "depressed", "panicked", "furious", "empathetic", "embarrassed", "reluctant", "disgusted", "keen", "moved", "proud", "relaxed", "grateful", "confident", "interested", "curious", "confused", "joyful", "disapproving", "negative", "denying", "astonished", "serious", "sarcastic", "conciliative", "comforting", "sincere", "sneering", "hesitating", "yielding", "painful", "awkward", "amused", "loving", "dating", "longing", "aroused", "seductive", "ecstatic", "shy"

---

## 特殊タグの挿入（text）

読み上げテキストについて、全角括弧を使った以下のような制御タグを利用できます。
ただしタグは自動アノテーションなので出現していないものもあると思われ、効果はものによると思われます。

### 1. 声の変化（スタイル・感情・意図）
- 感情/トーン：`（優しく）` `（囁き）` `（自信なさげに）` `（からかうように）` `（挑発するように）` `（独り言のように）`
- 感情の推移：`（徐々に怒りを込めて）` `（だんだん悲しげに）` `（喜びを爆発させて）`
- 声の状態：`（声が震えて）` `（眠そうに）` `（酔っ払って）` `（声を枯らして）`

### 2. 非言語的な発声
- 感情的な発声：`（うめき声）` `（吐息）` `（息切れ）` `（嗚咽）` `（くすくす笑い）` `（小さな悲鳴）`
- 呼吸：`（息をのむ）` `（深い溜息）` `（荒い息遣い）`
- 口の音：`（舌打ち）` `（リップノイズ）` `（唾を飲み込む音）`

### 3. アクション
- 話者の動作：`（笑いながら）` `（泣きながら）` `（咳き込みながら）` `（勢いに任せて攻撃）`
- 受ける動作：`（持ち上げられて）` `（首を絞められて）` `（腹を押しつぶされて）`

### 4. 音響・効果音
- 接触音：`（キス音）` `（耳舐め）` `（打撃音）` `（衣擦れの音）`
- NSFW関連音：`（チュパ音）` `（フェラ音）` `（ピストン音）` `（射精音）` `（粘着質な水音）`
- 環境音：`（ドアの開閉音）` `（足音）` `（雨音）`
- 音響効果：`（電話越しに）` `（スピーカー越しに）` `（エコー）`

### 5. 発話のリズム・間
- ペース：`（早口で）` `（ゆっくりと強調して）` `（一気にまくしたてて）`
- 間：`（少し間を置いて）` `（一呼吸おいて）` `（沈黙）`

### 6. 距離感・位置関係
- 位置：`（遠くから）` `（耳元で）` `（背後から）` `（ドア越しに）`

> 例：  
> `（囁き）ふふ…今日はよく頑張ったね。（キス音）`  
> `（徐々に怒りを込めて）もう一度言う。`

---

## 使い方（要点）

1. TTS タブでテキストとメタデータを入力（caption は必須）。  
2. 参照音声があれば読み込み（最大 15 秒、モノラル化・自動リサンプル）。参照音声とメタデータの組み合わせは未検証です。  
3. 例を使いたいときは Examples タブで選択→Apply to TTS。  
4. 「Synthesize」を押して音声を生成。
"""

with gr.Blocks() as app_readme:
    gr.Markdown(README_MD)

# ===============================
# UI: Credits Tab (unchanged)
# ===============================
with gr.Blocks() as app_credits:
    gr.Markdown("""
# Credits

* [zhenye234](https://github.com/zhenye234) for the original [repo](https://github.com/zhenye234/LLaSA_training)
* [mrfakename](https://huggingface.co/mrfakename) for the [gradio demo code](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [SunderAli17](https://huggingface.co/SunderAli17) for the [gradio demo code](https://huggingface.co/spaces/SunderAli17/llasa-3b-tts)
""")

# ===============================
# Root app with tabs
# ===============================
with gr.Blocks() as app:
    gr.Markdown(
        """
# Anime Llasa 3B Captions

This is a web UI for [Anime Llasa 3B Captions model](https://huggingface.co/NandemoGHS/Anime-Llasa-3B-Captions), aligned with your metadata-driven preprocessing.
If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 15s, and shortening your prompt.
"""
    )
    gr.TabbedInterface(
        [app_tts, app_examples, app_readme, app_credits],
        ["TTS", "Examples", "README", "Credits"]
    )

app.launch(inbrowser=True)
