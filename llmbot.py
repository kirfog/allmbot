#!/usr/bin/env python3
import logging
import re
import time
from types import SimpleNamespace

import num2words
import numpy as np
import sherpa_onnx
import sounddevice as sd
from dotenv import dotenv_values
from llama_cpp import Llama
from torch._C import device
from torch.package.package_importer import PackageImporter

config = SimpleNamespace(**dotenv_values(".env"))

logger = logging.getLogger()

llm = Llama(
    model_path=config.MODEL_LLM_PATH,
    n_ctx=2048,
    n_threads=8,
    n_threads_batch=8,
    verbose=False,
)


class Bot:
    def __init__(
        self,
        llm: Llama = llm,
    ):
        self.llm = llm

        self.tts_model = PackageImporter(config.MODEL_TTS_PATH).load_pickle(
            "tts_models", "model"
        )
        self.device = device("cpu")
        self.voice = "baya"
        self.tts_model.to(self.device)

        self.sample_rate = 16000
        self.sample_rate_speak = 48000

        self.rec = self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{config.MODEL_STT_PATH}/tokens.txt",
            encoder=f"{config.MODEL_STT_PATH}/encoder.onnx",
            decoder=f"{config.MODEL_STT_PATH}/decoder.onnx",
            joiner=f"{config.MODEL_STT_PATH}/joiner.onnx",
            num_threads=1,
            model_type="zipformer2",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=1.2,
        )

        self.stt_stream = self.recognizer.create_stream()

        self.history = config.PROMPT
        self.commands = {}

    def think(self, text: str) -> str:
        logger.warning(f"USER: {text}")
        self.history += f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        output = self.llm(
            self.history,
            max_tokens=1024,
            echo=False,
        )
        llm_text = output["choices"][0]["text"].strip()  # type: ignore

        match = re.search(r"\[CALL:\s*(\w+)(?:\((.*?)\))?,\s*(.*?)\]", llm_text)
        if match:
            cmd_name = match.group(1)
            cmd_arg = match.group(2)
            voice_text = match.group(3)
            if cmd_name in self.commands:
                action_result = self.commands[cmd_name](cmd_arg)
                if action_result:
                    llm_text = voice_text

        self.history += f"{llm_text}<|im_end|>\n"
        logger.warning(f"I: {llm_text}")

        if not llm_text:
            llm_text = config.NOANSWER
        return llm_text

    def speak(self, text: str) -> None:
        text = re.sub(
            r"\d+",
            lambda m: num2words.num2words(int(m.group(0)), lang=config.LANGUAGE),
            text,
        )

        chunks = re.split(r"(?<=[.!?]) +", text)

        for chunk in chunks:
            if not chunk.strip():
                continue

            try:
                audio_tensor = self.tts_model.apply_tts(
                    text=chunk,
                    speaker=self.voice,
                    sample_rate=self.sample_rate_speak,
                    put_accent=True,
                    put_yo=True,
                    put_stress_homo=True,
                    put_yo_homo=True,
                )
                audio_data = audio_tensor.numpy()
                sd.play(audio_data, self.sample_rate_speak)
                sd.wait()
            except Exception as e:
                logger.error(f"Speak error on chunk | {chunk} |: {e}")

    def listen(self) -> None:
        def audio_callback(indata, frames, time, status):

            rms = np.sqrt(np.mean(indata**2))
            volume_norm = rms * 100
            bars = int(volume_norm)
            print(f"\r|{'█' * bars}{'-' * (100 - bars)}| {volume_norm:.2f}", end="")

            samples = indata.copy().flatten()
            self.stt_stream.accept_waveform(self.sample_rate, samples)

        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=audio_callback,
            dtype="float32",
        ):
            while True:
                while self.recognizer.is_ready(self.stt_stream):
                    self.recognizer.decode_stream(self.stt_stream)

                if self.recognizer.is_endpoint(self.stt_stream):
                    result = self.recognizer.get_result(self.stt_stream)
                    if result:
                        thought = self.think(result)
                        self.speak(thought)
                        self.recognizer.reset(self.stt_stream)
                time.sleep(0.1)


class BotActor(Bot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commands = {"do_the_thing": self.do_the_thing}

    def do_the_thing(self, number: str):
        logger.warning(f"do_the_thing {number}")


if __name__ == "__main__":
    bot = BotActor()

    try:
        bot.listen()
    except KeyboardInterrupt:
        print("\nBot stopped.")
