package com.example.aquasegv2

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.example.aquasegv2.ImageUtils.scaleMask
// YoloFast Pipeline where we get the YOLO+SAM pipeline working
class YoloFastSam(
    context: Context,
    private val sam: FastSam,
    private val listener: Listener
    // we get Yolo's bounding box from Instance segmentation
) : InstanceSegmentation.InstanceSegmentationListener {

    interface Listener {
        // we extract the result, the preprocessing time,inference and postprocessing time
        fun onResults(
            results: List<SegmentationResult>,
            preProcessTime: Long,
            inferenceTime: Long,
            postProcessTime: Long
        )
        fun onError(error: String)
        fun onEmpty()
    }

//we get Yolo11n-Segmentor
    private val yoloSegmentator = InstanceSegmentation(
        context = context.applicationContext,
        modelPath = "yolo11n-seg-eggnoegg-float16.tflite",
        labelPath = "labels.txt",
        instanceSegmentationListener = this,
        message = { /*  */ }
    )



    private var inputBitmap: Bitmap? = null

    // Timing values
    private var yoloPre: Long = 0
    private var yoloInf: Long = 0
    private var yoloPost: Long = 0

    fun run(inputBitmap: Bitmap) {
        Log.d("YoloFastSam", "run() called")
        this.inputBitmap = inputBitmap
        yoloPre = 0
        yoloInf = 0
        yoloPost = 0
        yoloSegmentator.invoke(inputBitmap, withMask = false)
    }

    override fun onError(error: String) {
        Log.e("YoloFastSam", "onError: $error")
        listener.onError(error)
    }

    override fun onEmpty() {
        Log.d("YoloFastSam", "onEmpty() called")
        listener.onEmpty()
    }

    override fun onDetect(
        interfaceTime: Long,
        results: List<SegmentationResult>,
        preProcessTime: Long,
        postProcessTime: Long
    ) {
        // Save YOLO timings
        yoloPre = preProcessTime
        yoloInf = interfaceTime
        yoloPost = postProcessTime

        val bitmap = inputBitmap ?: run {
            listener.onError("Input bitmap not available")
            return
        }

        val confidenceThreshold = 0.3f
        val filteredResults = results.filter { it.box.cnf > confidenceThreshold }
        Log.d("YoloFastSam", "Filtered results: ${filteredResults.size}")

        if (filteredResults.isEmpty()) {
            listener.onEmpty()
            return
        }

        val finalResults = mutableListOf<SegmentationResult>()
        var pending = filteredResults.size

        // For timing FastSAM
        var fastSamPre = 0L
        var fastSamInf = 0L
        var fastSamPost = 0L
// we get the bounding box and crop for the segmentation
        for (segResult in filteredResults) {
            val box = segResult.box
            val left = (box.x1 * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
            val top = (box.y1 * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
            val right = (box.x2 * bitmap.width).toInt().coerceIn(left + 1, bitmap.width)
            val bottom = (box.y2 * bitmap.height).toInt().coerceIn(top + 1, bitmap.height)
            val cropWidth = right - left
            val cropHeight = bottom - top

            if (cropWidth <= 0 || cropHeight <= 0) {
                pending--
                if (pending == 0) {
                    listener.onResults(
                        finalResults,
                        yoloPre + fastSamPre,
                        yoloInf + fastSamInf,
                        yoloPost + fastSamPost
                    )
                }
                continue
            }

            val cropped = Bitmap.createBitmap(bitmap, left, top, cropWidth, cropHeight)
            val classId = box.cls
            val className = box.clsName

            // --- FastSAM timing ---

            val fastSamInfStart = System.currentTimeMillis()
            sam.run(cropped) { samResults ->
                val fastSamInfEnd = System.currentTimeMillis()
                val fastSamInfTime = fastSamInfEnd - fastSamInfStart
                fastSamInf += fastSamInfTime
                val fastSamPostStart = System.currentTimeMillis()

                if (!samResults.isNullOrEmpty()) {
                    val maskThreshold = 0.3f
                    val bestSamResult = samResults.maxByOrNull { samResult ->
                        val mask = samResult.mask.scaleMask(cropWidth, cropHeight)
                        mask.sumOf { row -> row.count { it > maskThreshold } }
                    }

                    if (bestSamResult != null) {
                        val mask = bestSamResult.mask.scaleMask(cropWidth, cropHeight)
                        val fullMask = Array(bitmap.height) { FloatArray(bitmap.width) { 0f } }
                        for (y in 0 until cropHeight) {
                            for (x in 0 until cropWidth) {
                                if (mask[y][x] > maskThreshold) {
                                    fullMask[top + y][left + x] = mask[y][x]
                                }
                            }
                        }
                        val adjustedBox = bestSamResult.box.copy(
                            x1 = (left + bestSamResult.box.x1 * cropWidth) / bitmap.width,
                            y1 = (top + bestSamResult.box.y1 * cropHeight) / bitmap.height,
                            x2 = (left + bestSamResult.box.x2 * cropWidth) / bitmap.width,
                            y2 = (top + bestSamResult.box.y2 * cropHeight) / bitmap.height,
                            cls = classId,
                            clsName = className
                        )
                        finalResults.add(
                            SegmentationResult(
                                box = adjustedBox,
                                mask = fullMask
                            )
                        )
                    }
                }
                val fastSamPostEnd = System.currentTimeMillis()
                fastSamPost += (fastSamPostEnd - fastSamPostStart)

                pending--
                if (pending == 0) {
                    listener.onResults(
                        finalResults,
                        yoloPre + fastSamPre,
                        yoloInf + fastSamInf,
                        yoloPost + fastSamPost
                    )
                }
            }
        }
    }

    override fun segment(crop: Bitmap): Bitmap {
        return crop
    }
}