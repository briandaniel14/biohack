import { useState, useEffect, useRef, useCallback } from 'react'
import { loadDatasetData, loadDatasetSummary, getFrameUrl, frameToTimestamp, saveRunName } from './data.js'
import HyperparamPanel from './HyperparamPanel.jsx'

const SPEEDS = [1, 2, 5, 10]
const VIEW_MODES = [
  { key: 'raw', label: 'Raw' },
  { key: 'mask', label: 'Mask' },
  { key: 'diagnostic', label: 'Diagnostic' },
]

export default function TuningPage({
  currentDataset,
  onNavigateResults,
  onNavigateUpload,
  pipelineRunning,
  pipelineStep,
  onRunPipeline,
}) {
  const [measurements, setMeasurements] = useState([])
  const [trackSummary, setTrackSummary] = useState([])
  const [summary, setSummary] = useState(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [viewMode, setViewMode] = useState('raw')
  const [loading, setLoading] = useState(true)
  const [loop, setLoop] = useState(false)
  const [imgError, setImgError] = useState(false)
  const [runName, setRunName] = useState('')

  const timerRef = useRef(null)
  const preloadedRef = useRef({})
  const loopRef = useRef(false)
  const paramsRef = useRef({})
  useEffect(() => { loopRef.current = loop }, [loop])

  const frameCount = summary?.frame_count || 100

  useEffect(() => {
    if (!currentDataset) return
    setLoading(true)
    setFrame(0)
    setPlaying(false)

    Promise.all([
      loadDatasetData(currentDataset.id),
      loadDatasetSummary(currentDataset.id),
    ]).then(([{ measurements: m, trackSummary: ts }, summ]) => {
      setMeasurements(m)
      setTrackSummary(ts)
      setSummary(summ)
      setLoading(false)
    })

    // Preload raw frames only (masks/diagnostics may not exist yet)
    const cache = {}
    const fc = 100
    for (let i = 0; i < fc; i++) {
      const img = new Image()
      img.src = getFrameUrl(currentDataset.id, i, 'raw')
      cache[`raw-${i}`] = img
    }
    preloadedRef.current = cache
  }, [currentDataset])

  // Clear image error when frame or viewMode changes
  useEffect(() => { setImgError(false) }, [frame, viewMode])

  useEffect(() => {
    if (playing) {
      const fps = 4 * speed
      timerRef.current = setInterval(() => {
        setFrame(f => {
          if (f >= frameCount - 1) {
            if (loopRef.current) return 0
            setPlaying(false)
            return f
          }
          return f + 1
        })
      }, 1000 / fps)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, speed, frameCount])

  const jumpToFrame = useCallback((f) => {
    setFrame(Math.max(0, Math.min(frameCount - 1, f)))
    setPlaying(false)
  }, [frameCount])

  const handleRunPipeline = useCallback(() => {
    if (!currentDataset || pipelineRunning) return
    onRunPipeline(currentDataset.id, paramsRef.current)
  }, [currentDataset, pipelineRunning, onRunPipeline])

  if (!currentDataset) return (
    <div className="flex-1 flex items-center justify-center text-gray-500">
      <div className="text-center">
        <p className="text-lg mb-2">No dataset selected</p>
        <button onClick={onNavigateUpload} className="text-sm text-blue-400 hover:text-blue-400">
          ← Go to Upload
        </button>
      </div>
    </div>
  )


  return (
    <div className="flex-1 flex flex-col lg:flex-row min-h-0">
      {/* LEFT: Image + Controls */}
      <div className="w-full lg:w-[42%] flex flex-col min-h-0 border-b lg:border-b-0 lg:border-r border-gray-800">
        {/* Image */}
        <div className="flex-1 min-h-0 bg-black flex items-center justify-center p-4">
          {loading ? (
            <div className="text-gray-500 text-sm">Loading frames...</div>
          ) : imgError ? (
            <div className="text-center text-gray-500">
              <p className="text-sm mb-1">No {viewMode} images available</p>
              <p className="text-xs text-gray-600">Run the pipeline first to generate masks & diagnostics</p>
            </div>
          ) : (
            <img
              src={getFrameUrl(currentDataset.id, frame, viewMode)}
              alt={`Frame ${frame}`}
              className="w-full max-h-full cursor-crosshair"
              style={{ imageRendering: viewMode === 'diagnostic' ? 'auto' : 'pixelated', aspectRatio: viewMode === 'diagnostic' ? 'auto' : '1 / 1' }}
              onError={() => setImgError(true)}
            />
          )}
        </div>

        {/* Controls Card */}
        <div className="flex-none border-t border-gray-800 bg-gray-900/80 p-3 sm:p-4 space-y-3">
          {/* Frame info */}
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <span>
              Frame <span className="font-mono font-semibold text-white">{frame}</span>
              <span className="text-gray-500"> / {frameCount - 1}</span>
              <span className="text-gray-500 ml-2">— {frameToTimestamp(frame)}</span>
            </span>
            <div className="flex items-center gap-1">
              {VIEW_MODES.map(m => (
                <button
                  key={m.key}
                  onClick={() => setViewMode(m.key)}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                    viewMode === m.key
                      ? 'bg-blue-700 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300'
                  }`}
                >{m.label}</button>
              ))}
            </div>
          </div>

          {/* Scrub bar */}
          <input
            type="range"
            min={0}
            max={frameCount - 1}
            value={frame}
            onChange={e => { setFrame(parseInt(e.target.value)); setPlaying(false) }}
            className="w-full accent-blue-500"
          />

          {/* Transport controls */}
          <div className="flex flex-wrap items-center gap-1.5">
            <button onClick={() => { setFrame(0); setPlaying(false) }}
              className="p-1.5 rounded-md bg-gray-800 hover:bg-gray-700 transition-colors" title="Restart">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M6 6h2v12H6zm3.5 6 8.5 6V6z"/></svg>
            </button>
            <button onClick={() => jumpToFrame(frame - 1)}
              className="p-1.5 rounded-md bg-gray-800 hover:bg-gray-700 transition-colors">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M15.41 16.59 10.83 12l4.58-4.59L14 6l-6 6 6 6z"/></svg>
            </button>
            <button onClick={() => setPlaying(!playing)}
              className="px-5 py-1.5 bg-blue-700 hover:bg-blue-600 rounded-md text-sm font-medium w-[72px] text-center transition-colors">
              {playing ? 'Pause' : 'Play'}
            </button>
            <button onClick={() => jumpToFrame(frame + 1)}
              className="p-1.5 rounded-md bg-gray-800 hover:bg-gray-700 transition-colors">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M8.59 16.59 13.17 12 8.59 7.41 10 6l6 6-6 6z"/></svg>
            </button>
            <button onClick={() => setLoop(!loop)}
              className={`p-1.5 rounded-md transition-colors ${loop ? 'bg-blue-700 text-white' : 'bg-gray-800 hover:bg-gray-700'}`}
              title="Loop">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M7 7h10v3l4-4-4-4v3H5v6h2zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2z"/></svg>
            </button>
            <div className="ml-auto flex items-center gap-1 text-xs text-gray-500">
              {SPEEDS.map(s => (
                <button
                  key={s}
                  onClick={() => setSpeed(s)}
                  className={`px-2 py-0.5 rounded-md transition-colors ${speed === s ? 'bg-blue-700 text-white' : 'bg-gray-800 hover:bg-gray-700 text-gray-400'}`}
                >{s}×</button>
              ))}
            </div>
          </div>

          {/* Dataset info */}
          <div className="rounded-lg bg-gray-800/60 px-3 py-2 mt-1">
            <p className="text-[10px] text-gray-500 uppercase tracking-wide">Dataset</p>
            <p className="text-sm font-medium text-white truncate">{currentDataset.name || currentDataset.id}</p>
          </div>
        </div>
      </div>

      {/* RIGHT: Hyperparameters + Actions */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="flex-1 min-h-0 overflow-hidden">
          <HyperparamPanel onChange={vals => { paramsRef.current = vals }} />
        </div>

        {/* Bottom actions */}
        <div className="flex-none p-3 sm:p-4 border-t border-gray-800 bg-gray-900/80 space-y-2">
          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="Name this run..."
              value={runName}
              onChange={e => setRunName(e.target.value)}
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2.5 text-sm leading-none text-gray-200 placeholder-gray-600 focus:border-blue-500 focus:outline-none transition-colors"
            />
            <button
              onClick={async () => {
                if (!runName.trim() || !currentDataset) return
                await saveRunName(currentDataset.id, runName.trim())
              }}
              disabled={!runName.trim()}
              className="px-3 py-2.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-xs font-medium text-gray-300 border border-gray-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed leading-none"
            >Save</button>
          </div>
          <button
            onClick={handleRunPipeline}
            disabled={pipelineRunning}
            className={`w-full py-2.5 rounded-lg text-sm font-semibold transition-colors disabled:cursor-not-allowed flex items-center justify-center gap-2 ${
              pipelineRunning
                ? 'bg-blue-700 opacity-40'
                : pipelineStep === 'Done!'
                  ? 'bg-green-600 text-white'
                  : 'bg-blue-700 hover:bg-blue-600'
            }`}
          >
            {pipelineRunning ? (
              <><svg className="inline-block w-4 h-4 animate-spin mr-1.5" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg>{pipelineStep}</>
            ) : pipelineStep === 'Done!' ? (
              <><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7"/></svg> Done!</>
            ) : (
              <><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg> Run Pipeline</>
            )}
          </button>
          <button
            onClick={async () => {
              if (runName.trim() && currentDataset) {
                await saveRunName(currentDataset.id, runName.trim())
              }
              onNavigateResults()
            }}
            className="w-full py-2.5 rounded-lg border border-gray-700 bg-gray-800 hover:bg-gray-700 text-sm font-medium transition-colors flex items-center justify-center gap-2"
          >
            View Results
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5-5 5M6 12h12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
