import { useState, useEffect, useRef, useCallback } from 'react'
import { loadDatasetData, loadDatasetSummary, getFrameUrl, frameToTimestamp } from './data.js'
import TrackInspector from './TrackInspector.jsx'
import MetricsDashboard from './MetricsDashboard.jsx'

const SPEEDS = [1, 2, 5, 10]
const VIEW_MODES = [
  { key: 'raw', label: 'Raw' },
  { key: 'mask', label: 'Mask' },
  { key: 'diagnostic', label: 'Diagnostic' },
]

export default function ResultsPage({
  currentDataset,
  onNavigateTuning,
}) {
  const [measurements, setMeasurements] = useState([])
  const [trackSummary, setTrackSummary] = useState([])
  const [summary, setSummary] = useState(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [viewMode, setViewMode] = useState('raw')
  const [selectedTrackId, setSelectedTrackId] = useState(null)
  const [loading, setLoading] = useState(true)
  const [loop, setLoop] = useState(false)
  const [imgError, setImgError] = useState(false)

  const timerRef = useRef(null)
  const preloadedRef = useRef({})
  const loopRef = useRef(false)
  useEffect(() => { loopRef.current = loop }, [loop])

  const frameCount = summary?.frame_count || 100

  useEffect(() => {
    if (!currentDataset) return
    setLoading(true)
    setFrame(0)
    setPlaying(false)
    setSelectedTrackId(null)

    Promise.all([
      loadDatasetData(currentDataset.id),
      loadDatasetSummary(currentDataset.id),
    ]).then(([{ measurements: m, trackSummary: ts }, summ]) => {
      setMeasurements(m)
      setTrackSummary(ts)
      setSummary(summ)
      setLoading(false)
    })

    const cache = {}
    const fc = 100
    for (let i = 0; i < fc; i++) {
      const img = new Image()
      img.src = getFrameUrl(currentDataset.id, i, 'raw')
      cache[`raw-${i}`] = img
    }
    preloadedRef.current = cache
  }, [currentDataset])

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

  const handleFrameClick = useCallback((e) => {
    if (viewMode !== 'raw') return
    const rect = e.target.getBoundingClientRect()
    const scaleX = 128 / rect.width
    const scaleY = 128 / rect.height
    const clickX = (e.clientX - rect.left) * scaleX
    const clickY = (e.clientY - rect.top) * scaleY

    const frameMeas = measurements.filter(m => m.frame === frame)
    let bestDist = Infinity
    let bestTrack = null

    for (const m of frameMeas) {
      if (clickY >= m.bbox_min_row && clickY <= m.bbox_max_row &&
          clickX >= m.bbox_min_col && clickX <= m.bbox_max_col) {
        const dist = Math.hypot(clickX - m.centroid_x, clickY - m.centroid_y)
        if (dist < bestDist) {
          bestDist = dist
          bestTrack = m.track_id
        }
      }
    }
    if (bestTrack != null) setSelectedTrackId(bestTrack)
  }, [measurements, frame, viewMode])

  const jumpToFrame = useCallback((f) => {
    setFrame(Math.max(0, Math.min(frameCount - 1, f)))
    setPlaying(false)
  }, [frameCount])

  if (!currentDataset) return (
    <div className="flex-1 flex items-center justify-center text-gray-500">
      <div className="text-center">
        <p className="text-lg mb-2">No dataset selected</p>
        <button onClick={onNavigateTuning} className="text-sm text-green-400 hover:text-green-400">← Go to Tuning</button>
      </div>
    </div>
  )

  return (
    <div className="flex-1 flex min-h-0">
      {/* LEFT: Image + Controls */}
      <div className="w-[42%] flex flex-col min-h-0 border-r border-gray-800">
        {/* Image */}
        <div className="flex-1 min-h-0 bg-black flex items-center justify-center p-4">
          {loading ? (
            <div className="text-gray-500 text-sm">Loading frames...</div>
          ) : imgError ? (
            <div className="text-center text-gray-500">
              <p className="text-sm mb-1">No {viewMode} images available</p>
              <p className="text-xs text-gray-600">Run the pipeline on the Tuning page first</p>
            </div>
          ) : (
            <img
              src={getFrameUrl(currentDataset.id, frame, viewMode)}
              alt={`Frame ${frame}`}
              className="w-full max-h-full cursor-crosshair"
              style={{ imageRendering: viewMode === 'diagnostic' ? 'auto' : 'pixelated', aspectRatio: viewMode === 'diagnostic' ? 'auto' : '1 / 1' }}
              onClick={handleFrameClick}
              onError={() => setImgError(true)}
            />
          )}
        </div>

        {/* Controls */}
        <div className="flex-none border-t border-gray-800 bg-gray-900/80 p-4 space-y-3">
          {/* Frame info */}
          <div className="flex items-center justify-between text-sm">
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
                      ? 'bg-green-700 text-white'
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
            className="w-full accent-green-500"
          />

          {/* Transport controls */}
          <div className="flex items-center gap-1.5">
            <button onClick={() => { setFrame(0); setPlaying(false) }}
              className="p-1.5 rounded-md bg-gray-800 hover:bg-gray-700 transition-colors" title="Restart">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M6 6h2v12H6zm3.5 6 8.5 6V6z"/></svg>
            </button>
            <button onClick={() => jumpToFrame(frame - 1)}
              className="p-1.5 rounded-md bg-gray-800 hover:bg-gray-700 transition-colors">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M15.41 16.59 10.83 12l4.58-4.59L14 6l-6 6 6 6z"/></svg>
            </button>
            <button onClick={() => setPlaying(!playing)}
              className="px-5 py-1.5 bg-green-700 hover:bg-green-600 rounded-md text-sm font-medium w-[72px] text-center transition-colors">
              {playing ? 'Pause' : 'Play'}
            </button>
            <button onClick={() => jumpToFrame(frame + 1)}
              className="p-1.5 rounded-md bg-gray-800 hover:bg-gray-700 transition-colors">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M8.59 16.59 13.17 12 8.59 7.41 10 6l6 6-6 6z"/></svg>
            </button>
            <button onClick={() => setLoop(!loop)}
              className={`p-1.5 rounded-md transition-colors ${loop ? 'bg-green-700 text-white' : 'bg-gray-800 hover:bg-gray-700'}`} title="Loop">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M7 7h10v3l4-4-4-4v3H5v6h2zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2z"/></svg>
            </button>
            <div className="ml-auto flex items-center gap-1 text-xs text-gray-500">
              {SPEEDS.map(s => (
                <button
                  key={s}
                  onClick={() => setSpeed(s)}
                  className={`px-2 py-0.5 rounded-md transition-colors ${speed === s ? 'bg-green-700 text-white' : 'bg-gray-800 hover:bg-gray-700 text-gray-400'}`}
                >{s}×</button>
              ))}
            </div>
          </div>


        </div>
      </div>

      {/* RIGHT: Track Inspector + Charts */}
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex-none border-b border-gray-800" style={{ height: '45%' }}>
          <TrackInspector
            trackSummary={trackSummary}
            measurements={measurements}
            selectedTrackId={selectedTrackId}
            onSelectTrack={setSelectedTrackId}
            onJumpToFrame={jumpToFrame}
            currentFrame={frame}
          />
        </div>
        <div className="flex-1 min-h-0">
          <MetricsDashboard
            measurements={measurements}
            trackSummary={trackSummary}
          />
        </div>
      </div>
    </div>
  )
}
