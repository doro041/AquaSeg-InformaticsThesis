package com.example.aquasegv2

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.example.aquasegv2.ImageUtils.scaleMask
import com.example.aquasegv2.MetaData.extractNamesFromMetadata
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class FastSam(
    context: Context,
    modelPath: String
) {
    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    private var xPoints = 0
    private var yPoints = 0
    private var masksNum = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(0f, 255f))
        .build()

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        labels.addAll(extractNamesFromMetadata(model))
        if (labels.isEmpty()) {
            labels.addAll(MetaData.TEMP_CLASSES)
        }

        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape0 = interpreter.getOutputTensor(0)?.shape()
        val outputShape1 = interpreter.getOutputTensor(1)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape0 != null) {
            numChannel = outputShape0[1]
            numElements = outputShape0[2]
        }

        if (outputShape1 != null) {
            if (outputShape1[1] == 32) {
                masksNum = outputShape1[1]
                xPoints = outputShape1[2]
                yPoints = outputShape1[3]
            } else {
                xPoints = outputShape1[1]
                yPoints = outputShape1[2]
                masksNum = outputShape1[3]
            }
        }
    }

    fun close() {
        interpreter.close()
    }

    fun run(
        frame: Bitmap,
        callback: (List<SegmentationResult>?) -> Unit
    ) {
        Log.d("FastSam", "FastSam run() called")
        if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0 ||
            xPoints == 0 || yPoints == 0 || masksNum == 0
        ) {
            callback(null)
            return
        }

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)
        val tensorImage = TensorImage(DataType.FLOAT32).apply {
            load(resizedBitmap)
        }
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = arrayOf(processedImage.buffer)

        val coordinatesBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, numChannel, numElements),
            DataType.FLOAT32
        )
        val maskProtoBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, xPoints, yPoints, masksNum),
            DataType.FLOAT32
        )
        val outputBuffer = mapOf(
            0 to coordinatesBuffer.buffer.rewind(),
            1 to maskProtoBuffer.buffer.rewind()
        )

        interpreter.runForMultipleInputsOutputs(imageBuffer, outputBuffer)
        Log.d("FastSam", "Inference done.")

        val bestBoxes = bestBox(coordinatesBuffer.floatArray) ?: run {
            callback(null)
            return
        }

        val maskProto = reshapeMaskOutput(maskProtoBuffer.floatArray)
        val segmentationResults = bestBoxes.map {
            SegmentationResult(
                box = it,
                mask = getFinalMask(frame.width, frame.height, it, maskProto)
            )
        }

        callback(segmentationResults)
    }

    fun runOnBox(
        bitmap: Bitmap,
        boundingBox: Output0,
        callback: (SegmentationResult?) -> Unit
    ) {
        val left = (boundingBox.x1 * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
        val top = (boundingBox.y1 * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
        val right = (boundingBox.x2 * bitmap.width).toInt().coerceIn(left + 1, bitmap.width)
        val bottom = (boundingBox.y2 * bitmap.height).toInt().coerceIn(top + 1, bitmap.height)
        val cropWidth = right - left
        val cropHeight = bottom - top

        if (cropWidth <= 0 || cropHeight <= 0) {
            callback(null)
            return
        }

        val cropped = Bitmap.createBitmap(bitmap, left, top, cropWidth, cropHeight)

        run(cropped) { results ->
            if (results.isNullOrEmpty()) {
                callback(null)
                return@run
            }
            val segResult = results.first()
            val mask = segResult.mask.scaleMask(cropWidth, cropHeight)
            val fullMask = Array(bitmap.height) { FloatArray(bitmap.width) { 0f } }

            for (y in 0 until cropHeight) {
                for (x in 0 until cropWidth) {
                    fullMask[top + y][left + x] = mask[y][x]
                }
            }

            val adjustedBox = segResult.box.copy(
                x1 = (left + segResult.box.x1 * cropWidth) / bitmap.width,
                y1 = (top + segResult.box.y1 * cropHeight) / bitmap.height,
                x2 = (left + segResult.box.x2 * cropWidth) / bitmap.width,
                y2 = (top + segResult.box.y2 * cropHeight) / bitmap.height
            )

            callback(
                SegmentationResult(
                    box = adjustedBox,
                    mask = fullMask
                )
            )
        }
    }

    private fun bestBox(array: FloatArray): List<Output0>? {
        val output0List = mutableListOf<Output0>()
        for (c in 0 until numElements) {
            var maxConf = CONFIDENCE_THRESHOLD
            var maxIdx = -1
            var currentInd = 4
            var arrayIdx = c + numElements * currentInd

            while (currentInd < (numChannel - masksNum)) {
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = currentInd - 4
                }
                currentInd++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                val clsName = labels.getOrElse(maxIdx) { "class$maxIdx" }
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w / 2F)
                val y1 = cy - (h / 2F)
                val x2 = cx + (w / 2F)
                val y2 = cy + (h / 2F)
                if (x1 < 0F || x2 > 1F || y1 < 0F || y2 > 1F) continue

                val maskWeight = mutableListOf<Float>()
                while (currentInd < numChannel) {
                    maskWeight.add(array[arrayIdx])
                    currentInd++
                    arrayIdx += numElements
                }

                output0List.add(
                    Output0(
                        x1, y1, x2, y2, cx, cy, w, h,
                        maxConf, maxIdx, clsName, maskWeight
                    )
                )
            }
        }

        if (output0List.isEmpty()) return null

        val nmsFiltered = applyNMS(output0List)
        return mutableListOf(selectCentralBox(nmsFiltered))
    }

    private fun applyNMS(output0List: List<Output0>): MutableList<Output0> {
        val sortedBoxes = output0List.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<Output0>()
        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.removeAt(0)
            selectedBoxes.add(first)
            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (calculateIoU(first, next) >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: Output0, box2: Output0): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val unionArea = box1.w * box1.h + box2.w * box2.h - intersectionArea
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    private fun selectCentralBox(boxes: List<Output0>): Output0 {
        val centerX = 0.5f
        val centerY = 0.5f
        return boxes.minByOrNull {
            val dx = it.cx - centerX
            val dy = it.cy - centerY
            dx * dx + dy * dy
        } ?: boxes.first()
    }

    private fun getFinalMask(
        width: Int,
        height: Int,
        output0: Output0,
        output1: List<Array<FloatArray>>
    ): Array<FloatArray> {
        val output1Copy = output1.map { it.map { it.copyOf() }.toTypedArray() }
        val relX1 = output0.x1 * xPoints
        val relY1 = output0.y1 * yPoints
        val relX2 = output0.x2 * xPoints
        val relY2 = output0.y2 * yPoints

        val result = Array(yPoints) { FloatArray(xPoints) }
        for ((index, proto) in output1Copy.withIndex()) {
            for (y in 0 until yPoints) {
                for (x in 0 until xPoints) {
                    proto[y][x] *= output0.maskWeight[index]
                    if (x + 1 > relX1 && x + 1 < relX2 && y + 1 > relY1 && y + 1 < relY2) {
                        result[y][x] += proto[y][x]
                    }
                }
            }
        }
        return result.scaleMask(width, height)
    }

    private fun reshapeMaskOutput(floatArray: FloatArray): List<Array<FloatArray>> {
        return List(masksNum) { mask ->
            Array(xPoints) { r ->
                FloatArray(yPoints) { c ->
                    floatArray[masksNum * yPoints * r + masksNum * c + mask]
                }
            }
        }
    }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.5F
        private const val IOU_THRESHOLD = 0.5F
    }
}




