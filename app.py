"""スプラトゥーン3 上達支援ツール「Splatool」

起動方法:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from splatool import ai_coach, knowledge
from splatool.match_log import STAGES, MatchLog
from splatool.providers import (
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    ProviderError,
)
from splatool.video_analyzer import extract_frames, select_frames_for_ai

# 単一系列チャート用のアクセントカラー(スプラのイカスミ+インク風)
ACCENT = "#7c3aed"

OFFLINE = "オフライン(定型アドバイス・無料)"

st.set_page_config(page_title="Splatool - スプラトゥーン3 上達支援", page_icon="🦑", layout="wide")

st.title("🦑 Splatool — スプラトゥーン3 上達支援ツール")


# --- サイドバー: AIバックエンド設定 -------------------------------------------
def build_provider():
    """サイドバーの選択に応じてAIプロバイダを構築する。オフラインなら None。"""
    st.header("⚙️ AIバックエンド")
    choice = st.radio(
        "使用するAI",
        [GeminiProvider.label, OllamaProvider.label, AnthropicProvider.label, OFFLINE],
        help="無料で使うなら Gemini(無料枠)か Ollama(ローカル)がおすすめ。",
    )

    provider = None
    if choice == GeminiProvider.label:
        key = st.text_input(
            "Gemini APIキー",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Google AI Studio (aistudio.google.com) で無料発行できます。",
        )
        st.caption(
            "🔑 [Google AI Studio](https://aistudio.google.com/apikey) で"
            "Googleアカウントがあれば**無料**でAPIキーを発行できます(クレジットカード不要)。"
            "無料枠にはレート制限があります。"
        )
        if key:
            provider = GeminiProvider(api_key=key)

    elif choice == OllamaProvider.label:
        host = st.text_input("Ollamaホスト", value="http://localhost:11434")
        model = st.text_input(
            "モデル名", value="llava", help="画像対応モデル: llava(軽量) / qwen2.5vl(高精度)"
        )
        st.caption(
            "💻 [ollama.com](https://ollama.com) からインストール後、"
            "`ollama pull llava` でモデルを取得。PCの性能で動く**完全無料**のAIです。"
        )
        provider = OllamaProvider(host=host, model=model)
        if not provider.available():
            st.warning("Ollamaに接続できません。アプリが起動しているか確認してください。")
            provider = None

    elif choice == AnthropicProvider.label:
        key = st.text_input(
            "Anthropic APIキー",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        st.caption("最も分析精度が高いですが、従量課金(有料)です。")
        if key:
            provider = AnthropicProvider(api_key=key)

    if provider is not None and provider.available():
        st.success(f"{choice} — 利用可能")
    else:
        st.info(
            "AI未接続 — オフライン機能(定型アドバイス・セルフレビュー・"
            "試合記録・知識ベース)はすべて無料で使えます。"
        )
        provider = None
    return provider


with st.sidebar:
    provider = build_provider()
    st.divider()
    st.caption(
        "動画はSwitchのキャプチャ(30秒)やキャプチャボードの録画(MP4)に対応。"
        "AI分析はフレームを間引いて送信するため、短いクリップほど精度が上がります。"
    )

log = MatchLog()

tab_video, tab_chat, tab_stats, tab_knowledge = st.tabs(
    ["🎬 動画分析", "🤖 AIコーチ相談", "📊 試合記録・統計", "📚 知識ベース"]
)


# --- 🎬 動画分析 --------------------------------------------------------------
with tab_video:
    st.subheader("プレイ動画の分析")
    st.markdown(
        "プレイ動画(MP4/MOV)をアップロードすると、フレームを抽出して"
        "**AIコーチのレポート**(AI接続時)または**セルフレビューガイド**(オフライン時)を表示します。"
    )

    uploaded = st.file_uploader("動画ファイル", type=["mp4", "mov", "avi", "mkv"])

    col1, col2 = st.columns(2)
    with col1:
        v_mode = st.selectbox("ルール(わかれば)", ["未選択"] + knowledge.MODES, key="v_mode")
        v_weapon = st.text_input("使用ブキ(わかれば)", placeholder="例: スプラシューター")
    with col2:
        v_class = st.selectbox(
            "ブキ種", ["未選択"] + knowledge.WEAPON_CLASSES, key="v_class"
        )
        v_concern = st.text_input(
            "特に見てほしいポイント", placeholder="例: すぐデスしてしまう / 打開が苦手"
        )

    if uploaded is not None:
        if st.button("分析する", type="primary"):
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded.name).suffix, delete=False
            ) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            try:
                with st.spinner("フレームを抽出中..."):
                    analysis = extract_frames(tmp_path)
                    max_send = provider.max_images if provider else 20
                    frames = select_frames_for_ai(analysis, max_send=max_send)

                st.success(
                    f"動画 {analysis.duration:.0f}秒 から {len(frames)}フレームを抽出 "
                    f"(注目シーン候補: {len(analysis.key_moments)}件)"
                )

                # 注目シーンのプレビュー
                key_frames = [f for f in frames if f.is_key_moment][:6]
                if key_frames:
                    st.markdown(
                        "**注目シーン候補**(画面変化が大きい瞬間 = キル/デス/スペシャル等の可能性)"
                    )
                    cols = st.columns(len(key_frames))
                    for c, f in zip(cols, key_frames):
                        c.image(f.jpeg_bytes, caption=f.label)

                context_lines = []
                if v_mode != "未選択":
                    context_lines.append(f"ルール: {v_mode}")
                if v_class != "未選択":
                    context_lines.append(f"ブキ種: {v_class}")
                if v_weapon:
                    context_lines.append(f"使用ブキ: {v_weapon}")
                if v_concern:
                    context_lines.append(f"プレイヤーの悩み: {v_concern}")
                context = "\n".join(context_lines) or "特になし"

                if provider is not None:
                    try:
                        with st.spinner("AIコーチが分析中...(1〜2分かかることがあります)"):
                            report = ai_coach.analyze_video_frames(provider, frames, context)
                        st.markdown("---")
                        st.markdown(report)
                    except ProviderError as e:
                        st.error(str(e))
                else:
                    # オフライン: セルフレビューガイド
                    st.markdown("---")
                    st.markdown("## 📝 セルフレビューガイド(無料・AIなし)")
                    st.markdown(
                        "上の**注目シーン候補**の時刻を動画で見返しながら、"
                        "当てはまる場面のチェック項目を確認してください。"
                        "デスの原因はほとんどがこのリストのどれかに当てはまります。"
                    )
                    for scene, checks in knowledge.SELF_REVIEW_CHECKLIST.items():
                        with st.expander(f"🔍 {scene}", expanded=(scene == "デスした場面")):
                            for c in checks:
                                st.checkbox(c, key=f"chk_{scene}_{c[:20]}")
                    st.markdown("### 選択したルール・ブキ種のアドバイス")
                    st.markdown(
                        knowledge.get_rule_based_advice(
                            v_mode if v_mode != "未選択" else None,
                            v_class if v_class != "未選択" else None,
                        )
                    )
            finally:
                os.unlink(tmp_path)


# --- 🤖 AIコーチ相談 ----------------------------------------------------------
with tab_chat:
    st.subheader("AIコーチに相談する")
    st.markdown("試合で困ったこと、立ち回りの疑問をなんでも聞いてください。")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("例: ガチエリアで打開がうまくいきません。ブキはスシです。"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if provider is not None:
                try:
                    with st.spinner("考え中..."):
                        reply = ai_coach.chat(provider, st.session_state.chat_history)
                except ProviderError as e:
                    reply = f"⚠️ {e}"
            else:
                # オフライン: キーワードからルール/ブキ種を推定して定型アドバイス
                mode = next(
                    (m for m in knowledge.MODES if m.replace("バトル", "") in prompt), None
                )
                wclass = next((w for w in knowledge.WEAPON_CLASSES if w in prompt), None)
                reply = (
                    "(オフライン版の回答です。サイドバーで無料のGemini APIキーを設定すると"
                    "AIが個別にアドバイスします)\n\n"
                    + knowledge.get_rule_based_advice(mode, wclass)
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.session_state.chat_history and st.button("会話をリセット"):
        st.session_state.chat_history = []
        st.rerun()


# --- 📊 試合記録・統計 --------------------------------------------------------
with tab_stats:
    st.subheader("試合を記録する")

    with st.form("match_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            m_mode = st.selectbox("ルール", knowledge.MODES)
            m_stage = st.selectbox("ステージ", STAGES)
            m_weapon = st.text_input("使用ブキ", placeholder="例: スプラシューター")
        with c2:
            m_result = st.radio("結果", ["WIN", "LOSE"], horizontal=True)
            m_kills = st.number_input("キル", 0, 50, 0)
            m_deaths = st.number_input("デス", 0, 50, 0)
        with c3:
            m_assists = st.number_input("アシスト", 0, 50, 0)
            m_specials = st.number_input("スペシャル使用回数", 0, 20, 0)
            m_paint = st.number_input("塗りポイント(任意)", 0, 5000, 0, step=50)
        m_memo = st.text_input("メモ(任意)", placeholder="例: 打開で焦って前に出すぎた")

        if st.form_submit_button("記録する", type="primary"):
            if not m_weapon.strip():
                st.error("使用ブキを入力してください")
            else:
                log.add(
                    mode=m_mode,
                    stage=m_stage,
                    weapon=m_weapon.strip(),
                    result=m_result,
                    kills=int(m_kills),
                    deaths=int(m_deaths),
                    assists=int(m_assists),
                    specials=int(m_specials),
                    paint_points=int(m_paint) or None,
                    memo=m_memo,
                )
                st.success("記録しました!")

    df = log.to_dataframe()
    st.divider()

    if df.empty:
        st.info("まだ記録がありません。試合結果を記録すると統計が表示されます。")
    else:
        st.subheader("統計")

        total = len(df)
        win_rate = df["win"].mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("総試合数", f"{total}")
        c2.metric("勝率", f"{win_rate:.1%}")
        c3.metric("平均キル", f"{df['kills'].mean():.1f}")
        c4.metric("平均デス", f"{df['deaths'].mean():.1f}")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**ルール別勝率**")
            mode_stats = df.groupby("mode")["win"].mean().sort_values(ascending=False)
            st.bar_chart(mode_stats, color=ACCENT, horizontal=True)
        with g2:
            st.markdown("**ブキ別勝率**(3戦以上)")
            weapon_grp = df.groupby("weapon").agg(n=("win", "size"), rate=("win", "mean"))
            weapon_stats = weapon_grp[weapon_grp["n"] >= 3]["rate"].sort_values(ascending=False)
            if weapon_stats.empty:
                st.caption("同じブキで3戦以上記録すると表示されます")
            else:
                st.bar_chart(weapon_stats, color=ACCENT, horizontal=True)

        st.markdown("**K/D推移**(直近50戦・時系列)")
        recent = df.sort_values("played_at").tail(50).reset_index(drop=True)
        kd_series = pd.Series(recent["kd"].values, name="K/D")
        st.line_chart(kd_series, color=ACCENT)

        if provider is not None and total >= 5:
            if st.button("🤖 AIに戦績の傾向を分析してもらう"):
                try:
                    with st.spinner("分析中..."):
                        st.markdown(ai_coach.analyze_match_stats(provider, log.summary_text()))
                except ProviderError as e:
                    st.error(str(e))
        elif total >= 5:
            st.caption("サイドバーでAIを設定すると、戦績データのAI傾向分析が使えます(Geminiなら無料)。")

        with st.expander("記録一覧・削除"):
            show_cols = [
                "id", "played_at", "mode", "stage", "weapon", "result",
                "kills", "deaths", "assists", "memo",
            ]
            st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
            del_id = st.number_input("削除するID", min_value=0, step=1, value=0)
            if st.button("削除") and del_id > 0:
                log.delete(int(del_id))
                st.rerun()


# --- 📚 知識ベース ------------------------------------------------------------
with tab_knowledge:
    st.subheader("立ち回り知識ベース(完全無料・オフライン)")

    k1, k2 = st.columns(2)
    with k1:
        st.markdown("### ルール別の立ち回り")
        sel_mode = st.selectbox("ルールを選択", knowledge.MODES, key="k_mode")
        for tip in knowledge.MODE_TIPS[sel_mode]:
            st.markdown(f"- {tip}")
    with k2:
        st.markdown("### ブキ種別のコツ")
        sel_class = st.selectbox("ブキ種を選択", knowledge.WEAPON_CLASSES, key="k_class")
        for tip in knowledge.WEAPON_TIPS[sel_class]:
            st.markdown(f"- {tip}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 共通の上達ポイント")
        for tip in knowledge.GENERAL_TIPS:
            st.markdown(f"- {tip}")
    with c2:
        st.markdown("### ギアの基本")
        for tip in knowledge.GEAR_TIPS:
            st.markdown(f"- {tip}")

    st.divider()
    st.markdown("### 📝 動画セルフレビューのチェックリスト")
    st.caption("AIを使わなくても、これを見ながら自分の動画を見返すだけで上達できます。")
    for scene, checks in knowledge.SELF_REVIEW_CHECKLIST.items():
        st.markdown(f"**{scene}**")
        for c in checks:
            st.markdown(f"- {c}")
