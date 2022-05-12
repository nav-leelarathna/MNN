package com.example.mnnconvolutionoptimisation

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import com.example.mnnconvolutionoptimisation.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Example of a call to a native method
//        binding.sampleText.text = profileNN()
//        binding.sampleText.text = profileKernel()
//        binding.sampleText.text = functionalityTest()
        binding.sampleText.text = benchmark()
//        binding.sampleText.text = deviceInformation()
//        binding.sampleText.text = testConvolution ()

    }

    /**
     * A native method that is implemented by the 'mnnconvolutionoptimisation' native library,
     * which is packaged with this application.
     */
//    external fun stringFromJNI(): String
    external fun profileKernel(): String
    external fun testConvolution(): String
    external fun functionalityTest(): String
    external fun benchmark(): String
    external fun deviceInformation(): String
    companion object {
        // Used to load the 'mnnconvolutionoptimisation' library on application startup.
        init {
            System.loadLibrary("mnnconvolutionoptimisation")
        }
    }
}

