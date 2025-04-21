package com.example.aquasegv2.ui.camera

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import android.widget.Toast
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider

import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.aquasegv2.*
import com.example.aquasegv2.databinding.FragmentCameraBinding

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment(), InstanceSegmentation.InstanceSegmentationListener {

    private lateinit var instanceSegmentation: InstanceSegmentation
    private lateinit var drawImages: DrawImages
    private lateinit var yoloFastSam: YoloFastSam
    private lateinit var fastSam: FastSam

    private var _binding: FragmentCameraBinding? = null
    private val binding get() = _binding!!

    private var segmentedBitmap: Bitmap? = null
    private var originalBitmap: Bitmap? = null
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private var useYoloPipeline = true

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentCameraBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                requireActivity(), REQUIRED_PERMISSIONS, 10
            )
        }

        binding.switchPipeline.setOnCheckedChangeListener { _, isChecked ->
            useYoloPipeline = !isChecked // Switch ON = FastSam, YOLO is false
            Log.d("CameraFragment", "Pipeline switched. useYoloPipeline = $useYoloPipeline (isChecked=$isChecked)")
            Toast.makeText(
                requireContext(),
                if (useYoloPipeline) "YOLO Pipeline" else "SamFAST Pipeline",
                Toast.LENGTH_SHORT
            ).show()

            clearTimingInfo()
        }

        drawImages = DrawImages(requireContext().applicationContext)

        // YOLO-only pipeline
        instanceSegmentation = InstanceSegmentation(
            context = requireContext().applicationContext,
            modelPath = "yolo11n-seg-eggnoegg-float16.tflite",
            labelPath = "labels.txt",
            instanceSegmentationListener = this,
            message = {
                Toast.makeText(requireContext().applicationContext, it, Toast.LENGTH_SHORT).show()
            }
        )

        fastSam = FastSam(
            context = requireContext().applicationContext,
            modelPath = "FastSAM-s_float16.tflite"
        )

        // YOLO+SAM pipeline
        yoloFastSam = YoloFastSam(
            context = requireContext().applicationContext,
            sam = fastSam,
            listener = object : YoloFastSam.Listener {
                override fun onResults(
                    results: List<SegmentationResult>,
                    preProcessTime: Long,
                    inferenceTime: Long,
                    postProcessTime: Long
                ) {
                    Log.d("CameraFragment", "YoloFastSam.Listener.onResults() called with ${results.size} results")
                    val image = drawImages.invoke(results)
                    if (!isAdded) return  // Check if fragment is attached
                    activity?.runOnUiThread {
                        if (!isAdded) return@runOnUiThread
                        segmentedBitmap = image
                        binding.ivTop.setImageBitmap(image)
                        binding.tvPreprocess.text = "$preProcessTime ms"
                        binding.tvInference.text = "$inferenceTime ms"
                        binding.tvPostprocess.text = "$postProcessTime ms"
                    }
                }
                override fun onError(error: String) {
                    Log.e("CameraFragment", "YoloFastSam.Listener.onError: $error")
                    if (!isAdded) return
                    activity?.runOnUiThread {
                        if (!isAdded) return@runOnUiThread
                        Toast.makeText(requireContext().applicationContext, error, Toast.LENGTH_SHORT).show()
                        clearTimingInfo()
                    }
                }
                override fun onEmpty() {
                    Log.d("CameraFragment", "YoloFastSam.Listener.onEmpty() called")
                    if (!isAdded) return
                    activity?.runOnUiThread {
                        if (!isAdded) return@runOnUiThread
                        segmentedBitmap = null
                        clearTimingInfo()
                    }
                }
            }
        )


        binding.captureButton.setOnClickListener {
            saveCombinedImage()
        }


        setupZoomSlider()

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()


            binding.zoomBar.progress = 0


            val preview = Preview.Builder()
                .setTargetResolution(Size(640, 640))
                .build()
                .also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setTargetResolution(Size(640, 640))
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(Executors.newSingleThreadExecutor(), ImageAnalyzer())
                }


            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {

                cameraProvider.unbindAll()

                camera = cameraProvider.bindToLifecycle(
                    viewLifecycleOwner, cameraSelector, preview, imageCapture, imageAnalyzer
                )

                // Apply zoom based on the current bar
                camera?.cameraControl?.setLinearZoom(binding.zoomBar.progress / 100f)

            } catch (exc: Exception) {
                Log.e("CameraFragment", "Camera binding failed: ${exc.message}")
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun setupZoomSlider() {
        binding.zoomBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                camera?.cameraControl?.setLinearZoom(progress / 100f)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun saveCombinedImage() {
        val original = originalBitmap ?: run {
            Toast.makeText(requireContext(), "No original frame to save.", Toast.LENGTH_SHORT).show()
            return
        }
        val segmented = segmentedBitmap ?: run {
            Toast.makeText(requireContext(), "No segmentation result to save.", Toast.LENGTH_SHORT).show()
            return
        }
        val combinedBitmap = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(combinedBitmap)
        canvas.drawBitmap(original, 0f, 0f, null)
        canvas.drawBitmap(segmented, 0f, 0f, null)
        saveToMediaStore(combinedBitmap)
    }

    private fun saveToMediaStore(bitmap: Bitmap) {
        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "combined_image_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/AquaSeg")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val resolver = requireContext().contentResolver
        val imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        if (imageUri != null) {
            resolver.openOutputStream(imageUri)?.use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            }
            contentValues.clear()
            contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(imageUri, contentValues, null, null)
            Toast.makeText(requireContext(), "Image saved to AquaSeg folder", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(requireContext(), "Failed to save image", Toast.LENGTH_SHORT).show()
        }
    }



    // YOLO-only pipeline results
    override fun onDetect(
        interfaceTime: Long,
        results: List<SegmentationResult>,
        preProcessTime: Long,
        postProcessTime: Long
    ) {
        if (!isAdded || activity == null || context == null) return

        if (results.isEmpty()) {
            Log.e("Segmentation", "No results detected!")
            requireActivity().runOnUiThread {
                if (!isAdded || context == null) return@runOnUiThread
                segmentedBitmap = null
                clearTimingInfo()
            }
            return
        }

        val image = drawImages.invoke(results)
        Log.d("Segmentation", "Segmentation successful, results applied to bitmap.")

        requireActivity().runOnUiThread {
            if (!isAdded || context == null) return@runOnUiThread
            segmentedBitmap = image
            binding.ivTop.setImageBitmap(image)
            binding.tvPreprocess.text = "$preProcessTime ms"
            binding.tvInference.text = "$interfaceTime ms"
            binding.tvPostprocess.text = "$postProcessTime ms"
        }
    }

    override fun segment(crop: Bitmap): Bitmap {

        return crop
    }

    override fun onEmpty() {
        if (!isAdded || activity == null) return
        activity?.runOnUiThread {
            binding.ivTop.setImageResource(0)
            clearTimingInfo()
        }
    }

    override fun onError(error: String) {
        requireActivity().runOnUiThread {
            Toast.makeText(requireContext().applicationContext, error, Toast.LENGTH_SHORT).show()
            clearTimingInfo()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        cameraExecutor.shutdown()
    }

    private fun clearTimingInfo() {
        binding.tvPreprocess.text = ""
        binding.tvInference.text = ""
        binding.tvPostprocess.text = ""
    }

    companion object {
        val REQUIRED_PERMISSIONS = mutableListOf(
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    inner class ImageAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            val bitmapBuffer = Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()
            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            }
            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )
            originalBitmap = rotatedBitmap // Save for saving images

            Log.d("ImageAnalyzer", "Analyzing frame. useYoloPipeline = $useYoloPipeline")
            if (useYoloPipeline) {
                Log.d("ImageAnalyzer", "YOLO Pipeline in use, calling instanceSegmentation.invoke()")
                instanceSegmentation.invoke(rotatedBitmap)
            } else {
                Log.d("ImageAnalyzer", "SamFAST Pipeline in use, calling yoloFastSam.run()")
                yoloFastSam.run(rotatedBitmap)
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            requireContext(),
            it
        ) == PackageManager.PERMISSION_GRANTED
    }
}