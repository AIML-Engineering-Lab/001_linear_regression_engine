import { useState, useEffect, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts'
import { Loader2, Github, BookOpen, AlertCircle, CheckCircle2 } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || '/api'
const REPO_NAME = '001_linear_regression_engine'
const GITHUB_URL = 'https://github.com/AIML-Engineering-Lab/001-linear-regression-engine'

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-brand-100 rounded-lg shadow-lg px-3 py-2 text-sm">
      <p className="font-semibold text-brand-800">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.fill }}>
          {p.name}: <span className="font-mono">{Number(p.value).toFixed(4)}</span>
        </p>
      ))}
    </div>
  )
}

// ─── Metric badge ─────────────────────────────────────────────────────────────
function MetricBadge({ label, value, highlight }) {
  return (
    <div className={`rounded-xl px-4 py-3 flex flex-col items-center
      ${highlight ? 'bg-brand-600 text-white' : 'bg-brand-50 text-brand-800'}`}>
      <span className="text-2xl font-bold font-mono">
        {typeof value === 'number' ? value.toFixed(4) : value}
      </span>
      <span className={`text-xs mt-1 uppercase tracking-wide font-semibold
        ${highlight ? 'text-brand-100' : 'text-brand-500'}`}>
        {label}
      </span>
    </div>
  )
}

// ─── Feature input ────────────────────────────────────────────────────────────
function FeatureInput({ name, value, onChange }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-semibold text-brand-700 uppercase tracking-wide">
        {name.replace(/_/g, ' ')}
      </label>
      <input
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(name, e.target.value)}
        className="w-full rounded-lg border border-brand-100 bg-white px-3 py-2
          text-sm font-mono text-brand-800
          focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent
          transition"
        placeholder="0.0"
      />
    </div>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [schema, setSchema]       = useState(null)
  const [modelMetrics, setModelMetrics] = useState(null)
  const [features, setFeatures]   = useState({})
  const [result, setResult]       = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')

  // Check API health + load schema on mount
  useEffect(() => {
    const init = async () => {
      try {
        const [healthRes, schemaRes, metricsRes] = await Promise.all([
          fetch(`${API_BASE}/health`),
          fetch(`${API_BASE}/schema`),
          fetch(`${API_BASE}/metrics`),
        ])
        if (!healthRes.ok) throw new Error('API offline')
        setApiStatus('online')
        const schemaData = await schemaRes.json()
        setSchema(schemaData)
        // Initialise all feature inputs to 0
        setFeatures(Object.fromEntries(schemaData.features.map((f) => [f, '0'])))
        if (metricsRes.ok) {
          setModelMetrics(await metricsRes.json())
        }
      } catch (e) {
        setApiStatus('offline')
        setError('API is offline. Start the FastAPI server first.')
      }
    }
    init()
  }, [])

  const handleFeatureChange = useCallback((name, val) => {
    setFeatures((prev) => ({ ...prev, [name]: val }))
  }, [])

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const payload = { features: Object.fromEntries(
        Object.entries(features).map(([k, v]) => [k, parseFloat(v) || 0])
      )}
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Prediction failed')
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Build bar chart data from feature values
  const chartData = schema
    ? schema.features.map((f) => ({
        name: f.length > 12 ? f.slice(0, 12) + '…' : f,
        value: parseFloat(features[f]) || 0,
      }))
    : []

  return (
    <div className="min-h-screen bg-gradient-to-br from-brand-50 to-white font-sans">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="bg-brand-800 text-white shadow-md">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <p className="text-brand-100 text-xs font-semibold uppercase tracking-widest mb-1">
              AIML Engineering Lab
            </p>
            <h1 className="text-xl font-bold">
              {REPO_NAME.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <span className={`flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full
              ${apiStatus === 'online' ? 'bg-green-500/20 text-green-300'
              : apiStatus === 'offline' ? 'bg-red-500/20 text-red-300'
              : 'bg-yellow-500/20 text-yellow-300'}`}>
              <span className={`w-2 h-2 rounded-full
                ${apiStatus === 'online' ? 'bg-green-400'
                : apiStatus === 'offline' ? 'bg-red-400'
                : 'bg-yellow-400 animate-pulse'}`} />
              API {apiStatus}
            </span>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer"
              className="flex items-center gap-1.5 text-xs text-brand-100 hover:text-white transition">
              <Github size={14} /> GitHub
            </a>
            <a href={`${GITHUB_URL}/blob/main/docs/PROJECT_REPORT.pdf`}
              target="_blank" rel="noreferrer"
              className="flex items-center gap-1.5 text-xs text-brand-100 hover:text-white transition">
              <BookOpen size={14} /> PDF Report
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 grid gap-6 lg:grid-cols-3">

        {/* ── Left panel: inputs ─────────────────────────────────────────── */}
        <div className="lg:col-span-1 flex flex-col gap-5">
          <div className="bg-white rounded-2xl shadow-sm border border-brand-100 p-5">
            <h2 className="text-sm font-bold text-brand-800 uppercase tracking-wide mb-4">
              Feature Inputs
              {schema && (
                <span className="ml-2 text-brand-400 font-normal normal-case tracking-normal">
                  ({schema.feature_count} features)
                </span>
              )}
            </h2>
            {!schema ? (
              <p className="text-brand-400 text-sm">Loading schema…</p>
            ) : (
              <div className="grid grid-cols-1 gap-3">
                {schema.features.map((f) => (
                  <FeatureInput
                    key={f}
                    name={f}
                    value={features[f] ?? '0'}
                    onChange={handleFeatureChange}
                  />
                ))}
              </div>
            )}
          </div>

          <button
            onClick={handlePredict}
            disabled={loading || apiStatus !== 'online' || !schema}
            className="w-full bg-brand-600 hover:bg-brand-700 disabled:bg-brand-100
              disabled:text-brand-400 text-white font-bold py-3 rounded-xl
              transition flex items-center justify-center gap-2 shadow-sm"
          >
            {loading
              ? <><Loader2 size={16} className="animate-spin" /> Predicting…</>
              : 'Run Prediction'}
          </button>

          {error && (
            <div className="flex items-start gap-2 bg-red-50 border border-red-100
              text-red-600 text-sm rounded-xl p-3">
              <AlertCircle size={16} className="mt-0.5 shrink-0" />
              {error}
            </div>
          )}
        </div>

        {/* ── Right panel: results ───────────────────────────────────────── */}
        <div className="lg:col-span-2 flex flex-col gap-5">

          {/* Prediction result */}
          {result && (
            <div className="bg-white rounded-2xl shadow-sm border border-brand-100 p-5">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle2 size={18} className="text-green-500" />
                <h2 className="text-sm font-bold text-brand-800 uppercase tracking-wide">
                  Prediction Result
                </h2>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <MetricBadge label="Prediction" value={result.prediction} highlight />
                {result.probability != null && (
                  <MetricBadge label="Confidence" value={result.probability} />
                )}
              </div>
            </div>
          )}

          {/* Model metrics */}
          {modelMetrics && !modelMetrics.message && (
            <div className="bg-white rounded-2xl shadow-sm border border-brand-100 p-5">
              <h2 className="text-sm font-bold text-brand-800 uppercase tracking-wide mb-4">
                Model Performance (Test Set)
              </h2>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {Object.entries(modelMetrics)
                  .filter(([, v]) => typeof v === 'number')
                  .map(([k, v]) => (
                    <MetricBadge key={k} label={k.replace(/_/g, ' ')} value={v} />
                  ))}
              </div>
            </div>
          )}

          {/* Feature visualisation */}
          {schema && chartData.length > 0 && (
            <div className="bg-white rounded-2xl shadow-sm border border-brand-100 p-5">
              <h2 className="text-sm font-bold text-brand-800 uppercase tracking-wide mb-4">
                Input Feature Values
              </h2>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EEF2FF" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine y={0} stroke="#6366F1" strokeDasharray="3 3" />
                  <Bar dataKey="value" fill="#6366F1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Empty state */}
          {!result && !error && (
            <div className="bg-brand-50 rounded-2xl border border-brand-100 p-8
              flex flex-col items-center justify-center text-center">
              <div className="text-4xl mb-3">🤖</div>
              <p className="text-brand-600 font-medium">
                Enter feature values and click <strong>Run Prediction</strong>
              </p>
              <p className="text-brand-400 text-sm mt-1">
                The model will run inference and display results here.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="mt-12 border-t border-brand-100 py-6">
        <p className="text-center text-xs text-brand-400">
          AIML Engineering Lab ·{' '}
          <a href="https://github.com/AIML-Engineering-Lab" className="underline hover:text-brand-600">
            github.com/AIML-Engineering-Lab
          </a>
          {' '}· Building intelligence, one project at a time.
        </p>
      </footer>
    </div>
  )
}
