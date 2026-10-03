from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import (
    QEvent, QEasingCurve, QObject, QPropertyAnimation, QRunnable, QThreadPool,
    QTimer, Qt, Signal, Slot,
)
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGraphicsOpacityEffect, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget,
)

from data_access import read_health, read_predictions, runtime_directory
from interaction import latest_delta, nearest_point_index, seconds_since


class WorkerSignals(QObject):
    completed = Signal(object)


class RuntimeReader(QRunnable):
    def __init__(self, include_predictions: bool, prediction_limit: int):
        super().__init__()
        self.include_predictions = include_predictions
        self.prediction_limit = prediction_limit
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        result: dict[str, Any] = {"health": None, "predictions": None, "errors": []}
        try:
            result["health"] = read_health()
        except Exception as error:
            result["errors"].append(f"Health file: {error}")
        if self.include_predictions:
            try:
                active_run = (result["health"] or {}).get("run_id")
                result["predictions"] = read_predictions(
                    limit=self.prediction_limit, run_id=active_run
                )
            except Exception as error:
                result["errors"].append(f"Prediction database: {error}")
        self.signals.completed.emit(result)


class UtcDateAxis(pg.DateAxisItem):
    def tickStrings(self, values, scale, spacing):
        if not values:
            return []
        if spacing >= 86400:
            pattern = "%d %b"
        elif spacing >= 3600:
            pattern = "%d %b %H:%M"
        else:
            pattern = "%H:%M"
        return [datetime.fromtimestamp(value, timezone.utc).strftime(pattern) for value in values]


class KpiCard(QFrame):
    def __init__(self, title: str, initial: str):
        super().__init__()
        self.setObjectName("kpiCard")
        self.setProperty("state", "normal")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setMinimumHeight(132)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 15, 18, 15)
        layout.setSpacing(7)
        heading = QLabel(title.upper())
        heading.setObjectName("kpiTitle")
        value_row = QHBoxLayout()
        value_row.setSpacing(10)
        self.value = QLabel(initial)
        self.value.setObjectName("kpiValue")
        self.value.setWordWrap(True)
        self.delta = QLabel("")
        self.delta.setObjectName("deltaBadge")
        self.delta.hide()
        value_row.addWidget(self.value)
        value_row.addWidget(self.delta)
        value_row.addStretch()
        self.detail = QLabel("")
        self.detail.setObjectName("kpiDetail")
        self.detail.setWordWrap(True)
        layout.addWidget(heading)
        layout.addLayout(value_row)
        layout.addWidget(self.detail)

    def update_value(self, value: str, detail: str = "") -> None:
        self.value.setText(value)
        self.detail.setText(detail)

    def update_delta(self, delta: float | None) -> None:
        if delta is None:
            self.delta.hide()
            return
        if delta > 0.0005:
            symbol, color = "↑", "#57c7ff"
        elif delta < -0.0005:
            symbol, color = "↓", "#f0b429"
        else:
            symbol, color = "→", "#8ea5ae"
        self.delta.setText(f"{symbol} Δ {delta:+.3f}")
        self.delta.setStyleSheet(
            f"color: {color}; background: #0b171d; border: 1px solid {color}; "
            "border-radius: 8px; padding: 3px 7px; font-weight: 600;"
        )
        self.delta.show()

    def set_state(self, state: str) -> None:
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)


class LegendSwatch(QWidget):
    def __init__(self, color: str, text: str, round_marker: bool = False):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        swatch = QFrame()
        swatch.setFixedSize(10 if round_marker else 18, 10 if round_marker else 3)
        swatch.setStyleSheet(
            f"background: {color}; border-radius: {5 if round_marker else 1}px;"
        )
        label = QLabel(text)
        label.setObjectName("legendText")
        layout.addWidget(swatch)
        layout.addWidget(label)


class Dashboard(QMainWindow):
    STATUS_COLORS = {
        "RUNNING": "#25d09f", "STARTING": "#f0b429", "CONNECTED": "#f0b429",
        "WARMING_UP": "#f0b429", "STALE": "#ef6461", "DATA_ERROR": "#ef6461",
        "DISCONNECTED": "#ef6461",
    }
    DEFAULT_HEARTBEAT_THRESHOLD_SECONDS = 90

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP Predictor Advisory Panel")
        self.resize(1280, 760)
        self.setMinimumSize(1020, 640)
        self.thread_pool = QThreadPool.globalInstance()
        self.worker_active = False
        self.poll_number = 0
        self.latest_rows: list[dict[str, Any]] = []
        self.plot_x: list[float] = []
        self.plot_y: list[float] = []
        self.plot_imputed: list[bool] = []
        self.pinned_index: int | None = None
        self.has_plotted_run: str | None = None
        self.active_run_id: str | None = None
        self.heartbeat_threshold_seconds: int | None = self.DEFAULT_HEARTBEAT_THRESHOLD_SECONDS
        self._build_ui()
        self._apply_theme()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.request_update)
        self.timer.start(2000)
        self.request_update(include_predictions=True)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(17)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("TSP Predictor Advisory Panel")
        title.setObjectName("mainTitle")
        subtitle = QLabel("SLURRY_FREE_ACID - read-only industrial advisory")
        subtitle.setObjectName("subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch()
        self.led = QFrame()
        self.led.setFixedSize(18, 18)
        self.led_effect = QGraphicsOpacityEffect(self.led)
        self.led.setGraphicsEffect(self.led_effect)
        self.pulse = QPropertyAnimation(self.led_effect, b"opacity", self)
        self.pulse.setDuration(1200)
        self.pulse.setStartValue(1.0)
        self.pulse.setKeyValueAt(0.5, 0.35)
        self.pulse.setEndValue(1.0)
        self.pulse.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.pulse.setLoopCount(-1)
        self.status_header = QLabel("STARTING")
        self.status_header.setObjectName("headerStatus")
        header.addWidget(self.led)
        header.addSpacing(8)
        header.addWidget(self.status_header)
        header.addSpacing(16)
        heartbeat_label = QLabel("HEARTBEAT")
        heartbeat_label.setObjectName("compactLabel")
        self.heartbeat_selector = QComboBox()
        for label, seconds in (("30 s", 30), ("90 s", 90), ("180 s", 180), ("Off", 0)):
            self.heartbeat_selector.addItem(label, seconds)
        self.heartbeat_selector.setCurrentIndex(1)
        self.heartbeat_selector.currentIndexChanged.connect(self._heartbeat_changed)
        self.heartbeat_selector.setToolTip(
            "Dashboard watchdog threshold. It does not control the plant or model."
        )
        header.addWidget(heartbeat_label)
        header.addWidget(self.heartbeat_selector)
        root.addLayout(header)

        content = QGridLayout()
        content.setHorizontalSpacing(18)
        content.setColumnStretch(0, 2)
        content.setColumnStretch(1, 5)
        self.prediction_card = KpiCard("Latest prediction", "--")
        self.status_card = KpiCard("Model status", "● STARTING")
        self.quality_card = KpiCard("Input quality", "… WAITING")
        cards = QVBoxLayout()
        cards.setSpacing(14)
        cards.addWidget(self.prediction_card)
        cards.addWidget(self.status_card)
        cards.addWidget(self.quality_card)
        cards.addStretch()
        content.addLayout(cards, 0, 0)

        chart_panel = QFrame()
        chart_panel.setObjectName("chartPanel")
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(18, 15, 18, 14)
        chart_header = QHBoxLayout()
        chart_title = QLabel("Prediction trend")
        chart_title.setObjectName("panelTitle")
        chart_header.addWidget(chart_title)
        chart_header.addSpacing(18)
        chart_header.addWidget(LegendSwatch("#2dd4a7", "Prediction / measured"))
        chart_header.addSpacing(10)
        chart_header.addWidget(LegendSwatch("#f0b429", "Imputed input", True))
        chart_header.addStretch()
        self.chart_count = QLabel("Waiting for predictions")
        self.chart_count.setObjectName("panelMeta")
        chart_header.addWidget(self.chart_count)
        chart_header.addSpacing(12)
        range_label = QLabel("SHOW")
        range_label.setObjectName("compactLabel")
        self.range_selector = QComboBox()
        for label, count in (("60 points", 60), ("360 points", 360), ("720 points", 720)):
            self.range_selector.addItem(label, count)
        self.range_selector.setCurrentIndex(2)
        self.range_selector.currentIndexChanged.connect(self._range_changed)
        self.clear_pin_button = QPushButton("Clear pin")
        self.clear_pin_button.setObjectName("secondaryButton")
        self.clear_pin_button.clicked.connect(self.clear_pin)
        self.clear_pin_button.hide()
        chart_header.addWidget(range_label)
        chart_header.addWidget(self.range_selector)
        chart_header.addWidget(self.clear_pin_button)
        chart_layout.addLayout(chart_header)

        self.plot = pg.PlotWidget(axisItems={"bottom": UtcDateAxis(orientation="bottom")})
        self.plot.setBackground("#0b171d")
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setLabel("left", "Predicted slurry free acid", units="%")
        self.plot.setLabel("bottom", "Process source timestamp (UTC)")
        for axis_name in ("left", "bottom"):
            self.plot.getAxis(axis_name).setTextPen("#a9bbc3")
            self.plot.getAxis(axis_name).setPen("#36505b")
        self.curve = self.plot.plot(pen=pg.mkPen("#2dd4a7", width=2.5))
        self.imputed_points = pg.ScatterPlotItem(
            pen=pg.mkPen("#f0b429", width=1), brush=pg.mkBrush("#f0b429"), size=7
        )
        self.plot.addItem(self.imputed_points)
        crosshair_pen = pg.mkPen("#79d9c0", width=1, style=Qt.PenStyle.DashLine)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        self.crosshair_v.hide()
        self.crosshair_h.hide()
        self.plot.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot.addItem(self.crosshair_h, ignoreBounds=True)
        self.hover_label = pg.TextItem(
            anchor=(0, 1), color="#ffffff", fill=pg.mkBrush(QColor(16, 33, 41, 235)),
            border=pg.mkPen("#36505b", width=1),
        )
        self.hover_label.hide()
        self.plot.addItem(self.hover_label, ignoreBounds=True)
        self.mouse_proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved
        )
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.plot.viewport().installEventFilter(self)
        chart_layout.addWidget(self.plot)
        content.addWidget(chart_panel, 0, 1)
        root.addLayout(content, stretch=1)

        footer = QHBoxLayout()
        self.message = QLabel("Starting dashboard...")
        self.message.setObjectName("footerText")
        runtime_label = QLabel(f"Runtime: {runtime_directory()}")
        runtime_label.setObjectName("footerText")
        footer.addWidget(self.message)
        footer.addStretch()
        footer.addWidget(runtime_label)
        root.addLayout(footer)
        self.setCentralWidget(central)

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #071116; color: #eef5f6; }
            QLabel#mainTitle { font-size: 27px; font-weight: 700; color: #ffffff; }
            QLabel#subtitle, QLabel#panelMeta, QLabel#footerText, QLabel#legendText {
                color: #8ea5ae; font-size: 12px;
            }
            QLabel#headerStatus { font-size: 14px; font-weight: 700; }
            QFrame#kpiCard, QFrame#chartPanel {
                background: #102129; border: 1px solid #24404a; border-radius: 12px;
            }
            QFrame#kpiCard[state="good"] { border-color: #27745f; }
            QFrame#kpiCard[state="warning"] { border-color: #9b721e; }
            QFrame#kpiCard[state="error"] { border-color: #9a4648; }
            QLabel#kpiTitle, QLabel#compactLabel {
                color: #7f9aa4; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
            }
            QLabel#kpiValue { color: #ffffff; font-size: 25px; font-weight: 700; }
            QLabel#kpiDetail { color: #9bb0b8; font-size: 11px; }
            QLabel#panelTitle { color: #ffffff; font-size: 17px; font-weight: 700; }
            QComboBox {
                background: #0b171d; color: #dbe7ea; border: 1px solid #36505b;
                border-radius: 6px; padding: 5px 9px; min-width: 88px;
            }
            QComboBox::drop-down { border: 0; width: 18px; }
            QComboBox QAbstractItemView {
                background: #102129; color: #eef5f6; selection-background-color: #24554d;
            }
            QPushButton#secondaryButton {
                background: #162a33; color: #b8cacf; border: 1px solid #36505b;
                border-radius: 6px; padding: 5px 9px;
            }
            QPushButton#secondaryButton:hover { border-color: #2dd4a7; color: #ffffff; }
        """)

    def request_update(self, include_predictions: bool = False) -> None:
        if self.worker_active:
            return
        self.poll_number += 1
        include_predictions = include_predictions or self.poll_number % 5 == 0
        self.worker_active = True
        worker = RuntimeReader(include_predictions, int(self.range_selector.currentData()))
        worker.signals.completed.connect(self.apply_runtime_data)
        self.thread_pool.start(worker)

    def _range_changed(self, _index: int | None = None) -> None:
        self.has_plotted_run = None
        self.request_update(include_predictions=True)

    def _heartbeat_changed(self, _index: int | None = None) -> None:
        value = int(self.heartbeat_selector.currentData())
        self.heartbeat_threshold_seconds = value or None
        self.request_update()

    @Slot(object)
    def apply_runtime_data(self, result: dict[str, Any]) -> None:
        self.worker_active = False
        health = result.get("health")
        if health:
            new_run = health.get("run_id")
            if new_run and new_run != self.active_run_id:
                self.active_run_id = new_run
                self.has_plotted_run = None
                self.clear_pin()
            self._update_health(health)
        rows = result.get("predictions")
        if rows is not None:
            self.latest_rows = rows
            self._update_predictions(rows)
        errors = result.get("errors", [])
        self.message.setText(" | ".join(errors) if errors else "Live files read successfully")

    def _update_health(self, health: dict[str, Any]) -> None:
        backend_status = str(health.get("status", "STARTING")).upper()
        age = seconds_since(health.get("updated_at"))
        stale = (
            self.heartbeat_threshold_seconds is not None
            and age is not None
            and age > self.heartbeat_threshold_seconds
            and backend_status != "STARTING"
        )
        status = "STALE" if stale else backend_status
        quality = str(health.get("input_quality", "WAITING")).upper()
        color = self.STATUS_COLORS.get(status, "#8ea5ae")
        self.led.setStyleSheet(
            f"background: {color}; border: 2px solid {QColor(color).lighter(150).name()}; "
            "border-radius: 9px;"
        )
        if status == "RUNNING" and self.pulse.state() != QPropertyAnimation.State.Running:
            self.pulse.start()
        elif status != "RUNNING":
            self.pulse.stop()
            self.led_effect.setOpacity(1.0)
        self.status_header.setText(status)
        icon = "●" if status == "RUNNING" else "◐" if status in {"STARTING", "CONNECTED", "WARMING_UP"} else "!"
        if age is None:
            freshness = "heartbeat time unavailable"
        elif self.heartbeat_threshold_seconds is None:
            freshness = f"heartbeat {int(age)}s ago · watchdog off"
        else:
            freshness = (
                f"heartbeat {int(age)}s ago · watchdog {self.heartbeat_threshold_seconds}s"
            )
        if status == "WARMING_UP":
            detail = (
                f"History {health.get('buffer_rows', 0)}/{health.get('rows_required', 31)} rows · "
                f"{freshness}"
            )
            if health.get("last_buffer_reset_reason"):
                detail += f" · reset {health.get('buffer_reset_count', 0)}: {health['last_buffer_reset_reason']}"
        elif status == "RUNNING":
            detail = f"Buffer ready: {health.get('buffer_rows', '--')} rows · {freshness}"
        elif status == "STALE":
            detail = f"No fresh service heartbeat · {freshness}"
        elif health.get("reasons"):
            detail = "; ".join(map(str, health["reasons"])) + f" · {freshness}"
        else:
            detail = str(health.get("message", freshness))
        self.status_card.update_value(f"{icon} {status}", detail)
        status_state = "good" if status == "RUNNING" else "warning" if status in {"STARTING", "CONNECTED", "WARMING_UP"} else "error"
        self.status_card.set_state(status_state)

        imputed = health.get("imputed_inputs", [])
        if quality == "GOOD":
            quality_value, quality_detail, quality_state = "✓ GOOD", "All required inputs available", "good"
        elif quality == "IMPUTED":
            quality_value = "⚠ IMPUTED"
            quality_detail = "Train-fitted imputation: " + ", ".join(imputed)
            quality_state = "warning"
        elif quality == "WAITING":
            quality_value, quality_detail, quality_state = "… WAITING", "Available once inference begins", "normal"
        else:
            quality_value, quality_detail, quality_state = f"! {quality}", "Input validation requires attention", "error"
        self.quality_card.update_value(quality_value, quality_detail)
        self.quality_card.set_state(quality_state)

    def _update_predictions(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            self.plot_x, self.plot_y, self.plot_imputed = [], [], []
            self.prediction_card.update_value("--", "Waiting for this run's first prediction")
            self.prediction_card.update_delta(None)
            self.chart_count.setText("Current run · no prediction yet")
            self.curve.setData([], [])
            self.imputed_points.setData([], [])
            self.clear_pin()
            return
        points = []
        for row in rows:
            try:
                stamp = datetime.fromisoformat(row["source_timestamp"].replace("Z", "+00:00"))
                points.append((stamp.timestamp(), float(row["predicted_slurry_free_acid"]), row))
            except (TypeError, ValueError, KeyError):
                continue
        points.sort(key=lambda item: item[0])
        if not points:
            return
        self.plot_x = [item[0] for item in points]
        self.plot_y = [item[1] for item in points]
        self.plot_imputed = [
            str(item[2].get("input_quality", "")).upper() == "IMPUTED" for item in points
        ]
        self.curve.setData(self.plot_x, self.plot_y)
        imputed = [item for item, flag in zip(points, self.plot_imputed) if flag]
        self.imputed_points.setData([p[0] for p in imputed], [p[1] for p in imputed])
        latest = points[-1]
        latest_time = datetime.fromtimestamp(latest[0], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.prediction_card.update_value(f"{latest[1]:.3f} %", f"Process time: {latest_time}")
        self.prediction_card.update_delta(latest_delta(self.plot_y))
        self.prediction_card.set_state("warning" if self.plot_imputed[-1] else "good")
        self.chart_count.setText(f"Current run · {len(points):,} points")
        run_id = str(latest[2].get("run_id", self.active_run_id or ""))
        if self.has_plotted_run != run_id:
            self.plot.enableAutoRange()
            self.has_plotted_run = run_id
        if self.pinned_index is not None:
            self.pinned_index = min(self.pinned_index, len(self.plot_x) - 1)
            self._show_point(self.pinned_index)

    def _mouse_moved(self, event) -> None:
        if self.pinned_index is not None or not self.plot_x:
            return
        scene_position = event[0]
        view_box = self.plot.plotItem.vb
        if not view_box.sceneBoundingRect().contains(scene_position):
            self._hide_inspector()
            return
        data_position = view_box.mapSceneToView(scene_position)
        index = nearest_point_index(self.plot_x, data_position.x())
        if index is not None:
            self._show_point(index)

    def _mouse_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self.plot_x:
            return
        view_box = self.plot.plotItem.vb
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        index = nearest_point_index(self.plot_x, view_box.mapSceneToView(event.scenePos()).x())
        if index is not None:
            self.pinned_index = index
            self.clear_pin_button.show()
            self._show_point(index, pinned=True)

    def _show_point(self, index: int, pinned: bool = False) -> None:
        x_value, y_value = self.plot_x[index], self.plot_y[index]
        imputed = self.plot_imputed[index]
        timestamp = datetime.fromtimestamp(x_value, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        quality_color = "#f0b429" if imputed else "#2dd4a7"
        quality_text = "⚠ imputed input" if imputed else "measured inputs"
        pin_text = " · pinned" if pinned or self.pinned_index is not None else ""
        self.hover_label.setHtml(
            f"<div style='padding:7px'><span style='color:#9bb0b8'>{timestamp}</span><br>"
            f"<span style='font-size:15px;color:#ffffff'><b>{y_value:.3f} %</b></span><br>"
            f"<span style='color:{quality_color}'>{quality_text}{pin_text}</span></div>"
        )
        self.crosshair_v.setPos(x_value)
        self.crosshair_h.setPos(y_value)
        x_range, y_range = self.plot.plotItem.vb.viewRange()
        self.hover_label.setAnchor((1 if x_value > sum(x_range) / 2 else 0,
                                    0 if y_value > sum(y_range) / 2 else 1))
        self.hover_label.setPos(x_value, y_value)
        self.crosshair_v.show()
        self.crosshair_h.show()
        self.hover_label.show()

    def _hide_inspector(self) -> None:
        if self.pinned_index is None:
            self.crosshair_v.hide()
            self.crosshair_h.hide()
            self.hover_label.hide()

    def clear_pin(self, _checked: bool = False) -> None:
        self.pinned_index = None
        self.clear_pin_button.hide()
        self.crosshair_v.hide()
        self.crosshair_h.hide()
        self.hover_label.hide()

    def eventFilter(self, watched, event):
        if watched is self.plot.viewport() and event.type() == QEvent.Type.Leave:
            self._hide_inspector()
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.thread_pool.waitForDone(1500)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("TSP Predictor Advisory Panel")
    app.setFont(QFont("Segoe UI", 10))
    pg.setConfigOptions(antialias=True)
    window = Dashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
