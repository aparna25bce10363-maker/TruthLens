import { useState } from "react"
import axios from "axios"
import { motion } from "framer-motion"

function App() {

  const [news, setNews] = useState("")
  const [result, setResult] = useState("")
  const [confidence, setConfidence] = useState("")
  const [loading, setLoading] = useState(false)

  const analyzeNews = async () => {

    if (!news) return

    setLoading(true)

    try {

      const response = await axios.post(
        "https://truthlens-giil.onrender.com/predict",
        {
          text: news
        }
      )

      setResult(response.data.prediction)

      setConfidence(response.data.confidence)

    } catch (error) {

      console.log(error)

      alert("Backend connection failed")

    } finally {

      setLoading(false)
    }
  }

  return (

    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">

      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}

        className="w-full max-w-3xl bg-white/10 border border-white/20 backdrop-blur-xl rounded-3xl p-8 shadow-2xl"
      >

        <h1 className="text-5xl font-bold text-center mb-4 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">

          TruthLens AI

        </h1>

        <p className="text-center text-slate-300 mb-8 text-lg">

          Futuristic Fake News Detection Platform

        </p>

        <textarea
          placeholder="Paste your news article here..."
          value={news}
          onChange={(e) => setNews(e.target.value)}

          className="w-full h-52 bg-slate-900/80 border border-cyan-500/30 rounded-2xl p-5 text-white text-lg outline-none focus:border-cyan-400 resize-none"
        />

        <button
          onClick={analyzeNews}

          className="mt-6 w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:scale-105 transition-all duration-300 text-white py-4 rounded-2xl text-xl font-semibold shadow-lg shadow-cyan-500/30"
        >

          {loading ? "Scanning..." : "Analyze News"}

        </button>

        {result && (

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}

            transition={{ duration: 0.5 }}

            className="mt-8 bg-slate-900/70 border border-white/10 rounded-2xl p-6 text-center"
          >

            <h2 className="text-3xl font-bold mb-3">

              Prediction:

              <span
                className={`ml-3 ${
                  result === "REAL"
                    ? "text-green-400"
                    : "text-red-400"
                }`}
              >

                {result}

              </span>

            </h2>

            <p className="text-slate-300 text-xl">

              Confidence Score:

              <span className="text-cyan-400 ml-2">

                {confidence}

              </span>

            </p>

          </motion.div>

        )}

      </motion.div>

    </div>
  )
}

export default App