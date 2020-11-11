package model.predictor

import javafx.scene.paint.Color

data class LabeledColor(
    val color: Color,
    val fontShade: FontShade
)

enum class FontShade(
    val color: Color,
    val intValue: Double,
    val outputArray: DoubleArray
) {
    DARK(Color.BLACK, 1.0, doubleArrayOf(0.0, 1.0)),
    LIGHT(Color.WHITE, 0.0, doubleArrayOf(1.0, 0.0))
}