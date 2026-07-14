"""AIバックエンドの抽象化レイヤー。

有料のClaude APIに加え、無料で使えるバックエンドを選択できる:
- GeminiProvider: Google AI Studioの無料枠APIキーで動作(画像分析対応)
- OllamaProvider: ローカルPCで動かす完全無料のLLM(要 Ollama インストール)

すべてのプロバイダは同じインターフェースを持つ:
    generate(system, messages, max_tokens) -> str
messages は Message(role, parts) のリスト。parts は ("text", 文字列) か
("image", base64エンコードJPEG) のタプル。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import requests

Part = tuple[Literal["text", "image"], str]


@dataclass
class Message:
    role: Literal["user", "assistant"]
    parts: list[Part]

    @classmethod
    def text(cls, role: Literal["user", "assistant"], text: str) -> "Message":
        return cls(role=role, parts=[("text", text)])


class ProviderError(RuntimeError):
    """ユーザーに表示できる日本語メッセージを持つエラー"""


class AnthropicProvider:
    """Claude API(高精度・有料)"""

    id = "claude"
    label = "Claude API(高精度・有料)"
    max_images = 20  # 動画分析でAIに送る最大フレーム数

    def __init__(self, api_key: str | None = None, model: str = "claude-opus-4-8"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, system: str, messages: list[Message], max_tokens: int = 16000) -> str:
        import anthropic

        api_messages = []
        for m in messages:
            content: list[dict] = []
            for kind, val in m.parts:
                if kind == "text":
                    content.append({"type": "text", "text": val})
                else:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": val,
                            },
                        }
                    )
            api_messages.append({"role": m.role, "content": content})

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            with client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                system=system,
                messages=api_messages,
            ) as stream:
                response = stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise ProviderError("Claude APIキーが無効です。キーを確認してください。") from e
        except anthropic.RateLimitError as e:
            raise ProviderError("Claude APIのレート制限に達しました。少し待ってから再試行してください。") from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"Claude APIエラー ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("Claude APIに接続できません。ネットワークを確認してください。") from e

        return "".join(b.text for b in response.content if b.type == "text")


class GeminiProvider:
    """Gemini API — Google AI Studioの無料枠で使える(レート制限あり)"""

    id = "gemini"
    label = "Gemini API(無料枠あり)"
    max_images = 16

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, system: str, messages: list[Message], max_tokens: int = 16000) -> str:
        contents = []
        for m in messages:
            parts: list[dict] = []
            for kind, val in m.parts:
                if kind == "text":
                    parts.append({"text": val})
                else:
                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": val}})
            contents.append({"role": "user" if m.role == "user" else "model", "parts": parts})

        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        try:
            r = requests.post(
                self.ENDPOINT.format(model=self.model),
                params={"key": self.api_key},
                json=body,
                timeout=300,
            )
        except requests.RequestException as e:
            raise ProviderError("Gemini APIに接続できません。ネットワークを確認してください。") from e

        if r.status_code == 429:
            raise ProviderError(
                "Gemini APIの無料枠レート制限に達しました。1分ほど待ってから再試行してください。"
            )
        if r.status_code in (401, 403):
            raise ProviderError("Gemini APIキーが無効です。Google AI Studioで発行したキーか確認してください。")
        if not r.ok:
            raise ProviderError(f"Gemini APIエラー ({r.status_code}): {r.text[:300]}")

        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Gemini APIから予期しない応答が返りました: {str(data)[:300]}") from e
        if not text:
            raise ProviderError("Gemini APIの応答が空でした。動画が長すぎる場合は短いクリップで試してください。")
        return text


class OllamaProvider:
    """Ollama — ローカルPCで動かす完全無料のLLM。

    事前に https://ollama.com からインストールし、画像対応モデルを取得しておく:
        ollama pull llava        (軽量・8GB RAM目安)
        ollama pull qwen2.5vl    (高精度・16GB RAM目安)
    """

    id = "ollama"
    label = "Ollama(ローカル・完全無料)"
    max_images = 6  # ローカルモデルは大量の画像が苦手なので絞る

    def __init__(self, host: str | None = None, model: str = "llava"):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model

    def available(self) -> bool:
        try:
            return requests.get(f"{self.host}/api/tags", timeout=2).ok
        except requests.RequestException:
            return False

    def generate(self, system: str, messages: list[Message], max_tokens: int = 16000) -> str:
        api_messages: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            text = "\n".join(val for kind, val in m.parts if kind == "text")
            images = [val for kind, val in m.parts if kind == "image"]
            msg: dict = {"role": m.role, "content": text}
            if images:
                msg["images"] = images
            api_messages.append(msg)

        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": api_messages, "stream": False},
                timeout=600,
            )
        except requests.RequestException as e:
            raise ProviderError(
                f"Ollamaに接続できません({self.host})。"
                "Ollamaアプリが起動しているか確認してください。"
            ) from e

        if r.status_code == 404:
            raise ProviderError(
                f"モデル「{self.model}」が見つかりません。"
                f"ターミナルで `ollama pull {self.model}` を実行してください。"
            )
        if not r.ok:
            raise ProviderError(f"Ollamaエラー ({r.status_code}): {r.text[:300]}")

        content = r.json().get("message", {}).get("content", "")
        if not content:
            raise ProviderError("Ollamaの応答が空でした。より小さい動画・少ない画像で試してください。")
        return content
