"""Claude APIを使ったプレー分析・コーチング機能。

ANTHROPIC_API_KEY が設定されていれば動画フレームのAI分析とチャット相談が使える。
未設定の場合、呼び出し側は knowledge.get_rule_based_advice() にフォールバックする。
"""

from __future__ import annotations

import os

import anthropic

from .video_analyzer import Frame

MODEL = "claude-opus-4-8"

COACH_SYSTEM_PROMPT = """\
あなたはスプラトゥーン3の上級コーチです。プレイヤーの上達を支援します。

方針:
- 日本語で回答する。ゲーム内用語(打開、抑え、潜伏、対面、カンモン、イカランプなど)は正しく使う。
- 指摘は具体的に。「立ち回りを意識しよう」ではなく「人数不利の場面で前に出ていたので、イカランプを見て2落ち以上なら一度下がる」のように行動レベルで伝える。
- 良かった点も必ず挙げる。ダメ出しだけのコーチングはしない。
- 改善点は優先度順に最大3つに絞る。一度に全部直そうとさせない。
- 断定できないこと(画面から読み取れないこと)は推測と明示する。
"""

VIDEO_ANALYSIS_PROMPT = """\
これはスプラトゥーン3のプレイ動画から抽出したフレームです。各画像の直前に動画内のタイムスタンプを記載しています。

以下の情報も参考にしてください:
{context}

フレームを時系列で分析し、次の形式でコーチングレポートを作成してください:

## 試合の概要
(ルール・ステージ・使用ブキ・大まかな試合展開を画面から読み取れる範囲で)

## 良かったプレー
(タイムスタンプを引用しつつ2〜3点)

## 改善ポイント(優先度順)
(タイムスタンプを引用しつつ最大3点。それぞれ「何が起きたか→なぜ問題か→次からどうするか」の順で)

## 次の練習メニュー
(改善ポイントに対応した具体的な練習方法を1〜2個)

注意: フレームは間引きされているため、映っていない時間帯があります。読み取れないことは無理に断定しないでください。
"""


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def analyze_video_frames(frames: list[Frame], context: str = "特になし") -> str:
    """抽出フレームをClaudeに送り、コーチングレポートを生成する。"""
    content: list[dict] = []
    for f in frames:
        content.append({"type": "text", "text": f"[タイムスタンプ {f.label}]"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": f.b64,
                },
            }
        )
    content.append({"type": "text", "text": VIDEO_ANALYSIS_PROMPT.format(context=context)})

    with _client().messages.stream(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=COACH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        response = stream.get_final_message()

    return "".join(b.text for b in response.content if b.type == "text")


def chat(history: list[dict[str, str]]) -> str:
    """コーチとのチャット。history は [{"role": "user"|"assistant", "content": str}, ...]"""
    with _client().messages.stream(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=COACH_SYSTEM_PROMPT,
        messages=history,
    ) as stream:
        response = stream.get_final_message()

    return "".join(b.text for b in response.content if b.type == "text")


def analyze_match_stats(stats_summary: str) -> str:
    """試合記録の集計結果をもとに傾向分析とアドバイスを生成する。"""
    prompt = f"""\
以下は私のスプラトゥーン3の戦績データの集計です。

{stats_summary}

このデータから読み取れる傾向(得意/苦手なルール・ステージ・ブキ、デス数の傾向など)を分析し、
勝率を上げるための具体的なアドバイスを優先度順に3つ提案してください。
"""
    with _client().messages.stream(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=COACH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    return "".join(b.text for b in response.content if b.type == "text")
