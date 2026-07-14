"""試合記録の保存と集計(SQLite)。"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "matches.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    played_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    stage TEXT NOT NULL,
    weapon TEXT NOT NULL,
    result TEXT NOT NULL CHECK (result IN ('WIN', 'LOSE')),
    kills INTEGER NOT NULL DEFAULT 0,
    deaths INTEGER NOT NULL DEFAULT 0,
    assists INTEGER NOT NULL DEFAULT 0,
    specials INTEGER NOT NULL DEFAULT 0,
    paint_points INTEGER,
    memo TEXT
);
"""

STAGES = [
    "ユノハナ大渓谷",
    "ゴンズイ地区",
    "ヤガラ市場",
    "マテガイ放水路",
    "ナメロウ金属",
    "マサバ海峡大橋",
    "キンメダイ美術館",
    "マヒマヒリゾート&スパ",
    "海女美術大学",
    "チョウザメ造船",
    "ザトウマーケット",
    "スメーシーワールド",
    "クサヤ温泉",
    "ヒラメが丘団地",
    "ナンプラー遺跡",
    "マンタマリア号",
    "タラポートショッピングパーク",
    "コンブトラック",
    "タカアシ経済特区",
    "オヒョウ海運",
    "バイガイ亭",
    "ネギトロ炭鉱",
    "カジキ空港",
    "リュウグウターミナル",
    "グランドバンカラアリーナ",
    "デカライン高架下",
    "その他",
]


class MatchLog:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add(
        self,
        mode: str,
        stage: str,
        weapon: str,
        result: str,
        kills: int = 0,
        deaths: int = 0,
        assists: int = 0,
        specials: int = 0,
        paint_points: int | None = None,
        memo: str = "",
        played_at: datetime | None = None,
    ) -> None:
        played_at = played_at or datetime.now()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO matches
                   (played_at, mode, stage, weapon, result, kills, deaths,
                    assists, specials, paint_points, memo)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    played_at.isoformat(timespec="seconds"),
                    mode,
                    stage,
                    weapon,
                    result,
                    kills,
                    deaths,
                    assists,
                    specials,
                    paint_points,
                    memo,
                ),
            )

    def delete(self, match_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM matches WHERE id = ?", (match_id,))

    def to_dataframe(self) -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM matches ORDER BY played_at DESC", conn
            )
        if not df.empty:
            df["played_at"] = pd.to_datetime(df["played_at"])
            df["win"] = (df["result"] == "WIN").astype(int)
            df["kd"] = df.apply(
                lambda r: r["kills"] / r["deaths"] if r["deaths"] > 0 else float(r["kills"]),
                axis=1,
            )
        return df

    def summary_text(self) -> str:
        """AI分析に渡すためのテキスト集計。"""
        df = self.to_dataframe()
        if df.empty:
            return "記録なし"

        lines = [
            f"総試合数: {len(df)}",
            f"勝率: {df['win'].mean():.1%}",
            f"平均キル: {df['kills'].mean():.1f} / 平均デス: {df['deaths'].mean():.1f}"
            f" / 平均アシスト: {df['assists'].mean():.1f}",
        ]
        for col, name in (("mode", "ルール"), ("weapon", "ブキ"), ("stage", "ステージ")):
            grp = (
                df.groupby(col)
                .agg(試合数=("win", "size"), 勝率=("win", "mean"), 平均デス=("deaths", "mean"))
                .sort_values("試合数", ascending=False)
            )
            lines.append(f"\n【{name}別】")
            for idx, row in grp.head(8).iterrows():
                lines.append(
                    f"- {idx}: {int(row['試合数'])}戦 勝率{row['勝率']:.0%} 平均デス{row['平均デス']:.1f}"
                )
        return "\n".join(lines)
