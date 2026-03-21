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

function getDownloadFilename(contentDisposition, fallbackName) {
  if (!contentDisposition) return fallbackName

  const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i)
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1])
    } catch {
      return utf8Match[1]
    }
  }

  const quotedMatch = contentDisposition.match(/filename="([^"]+)"/i)
  if (quotedMatch?.[1]) return quotedMatch[1]

  const bareMatch = contentDisposition.match(/filename=([^;]+)/i)
  if (bareMatch?.[1]) return bareMatch[1].trim()

  return fallbackName
}

export default function ResultsPage({
  currentDataset,
  onNavigateTuning,
}) {
  const [rows, setRows] = useState([])
  const [filamentSummary, setFilamentSummary] = useState([])
  const [summary, setSummary] = useState(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [viewMode, setViewMode] = useState('raw')
  const [selectedFilamentId, setSelectedFilamentId] = useState(null)
  const [loading, setLoading] = useState(true)
  const [loop, setLoop] = useState(false)
  const [imgError, setImgError] = useState(false)
  const [downloading, setDownloading] = useState(null)
  const [filteredFilamentIds, setFilteredFilamentIds] = useState(null)

  const timerRef = useRef(null)
  const preloadedRef = useRef({})
  const loopRef = useRef(false)
  const chartsRef = useRef(null)
  useEffect(() => { loopRef.current = loop }, [loop])

  const frameCount = summary?.frame_count || 100

  useEffect(() => {
    if (!currentDataset) return
    setLoading(true)
    setFrame(0)
    setPlaying(false)
    setSelectedFilamentId(null)

    Promise.all([
      loadDatasetData(currentDataset.id),
      loadDatasetSummary(currentDataset.id),
    ]).then(([{ rows: r, filamentSummary: fs }, summ]) => {
      setRows(r)
      setFilamentSummary(fs)
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

    const frameRows = rows.filter(r => r.frame === frame)
    let bestDist = Infinity
    let bestFilament = null

    for (const r of frameRows) {
      if (clickY >= r.bbox_min_row && clickY <= r.bbox_max_row &&
          clickX >= r.bbox_min_col && clickX <= r.bbox_max_col) {
        const dist = Math.hypot(clickX - r.centroid_x, clickY - r.centroid_y)
        if (dist < bestDist) {
          bestDist = dist
          bestFilament = r.filament_ID
        }
      }
    }
    if (bestFilament != null) setSelectedFilamentId(bestFilament)
  }, [rows, frame, viewMode])

  const jumpToFrame = useCallback((f) => {
    setFrame(Math.max(0, Math.min(frameCount - 1, f)))
    setPlaying(false)
  }, [frameCount])


  const svgToPng = useCallback((svgEl) => {
    return new Promise((resolve) => {
      const rect = svgEl.getBoundingClientRect()
      const w = Math.round(rect.width)
      const h = Math.round(rect.height)
      const clone = svgEl.cloneNode(true)
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
      clone.setAttribute('width', w)
      clone.setAttribute('height', h)
      if (!clone.getAttribute('viewBox')) clone.setAttribute('viewBox', '0 0 ' + w + ' ' + h)
      const svgStr = new XMLSerializer().serializeToString(clone)
      const dataUrl = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgStr)))
      const canvas = document.createElement('canvas')
      const scale = 2
      canvas.width = w * scale
      canvas.height = h * scale
      const ctx = canvas.getContext('2d')
      ctx.scale(scale, scale)
      const img = new Image()
      img.onload = () => {
        ctx.fillStyle = '#111827'
        ctx.fillRect(0, 0, w, h)
        ctx.drawImage(img, 0, 0, w, h)
        canvas.toBlob(resolve, 'image/png')
      }
      img.onerror = () => resolve(null)
      img.src = dataUrl
    })
  }, [])

  const captureScreenshots = useCallback(async () => {
    const screenshots = []

    // Capture chart SVGs with titles
    if (chartsRef.current) {
      const panels = chartsRef.current.querySelectorAll('.bg-gray-900.rounded')
      const chartNames = ['filament_length', 'filaments_per_frame', 'filament_area', 'filament_eccentricity']
      for (let i = 0; i < panels.length; i++) {
        const panel = panels[i]
        const titleEl = panel.querySelector('div')
        const title = titleEl?.textContent || ''
        const svgEl = panel.querySelector('svg.recharts-surface')
        if (!svgEl) continue

        const blob = await new Promise((resolve) => {
          const rect = svgEl.getBoundingClientRect()
          const w = Math.round(rect.width)
          const h = Math.round(rect.height)
          const titleH = 28
          const clone = svgEl.cloneNode(true)
          clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
          clone.setAttribute('width', w)
          clone.setAttribute('height', h)
          if (!clone.getAttribute('viewBox')) clone.setAttribute('viewBox', '0 0 ' + w + ' ' + h)
          const svgStr = new XMLSerializer().serializeToString(clone)
          const dataUrl = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgStr)))
          const canvas = document.createElement('canvas')
          const scale = 2
          canvas.width = w * scale
          canvas.height = (h + titleH) * scale
          const ctx = canvas.getContext('2d')
          ctx.scale(scale, scale)
          const img = new Image()
          img.onload = () => {
            ctx.fillStyle = '#111827'
            ctx.fillRect(0, 0, w, h + titleH)
            // Draw title
            ctx.fillStyle = '#ffffff'
            ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, sans-serif'
            ctx.textBaseline = 'middle'
            ctx.fillText(title.toUpperCase(), 8, titleH / 2)
            // Draw chart below title
            ctx.drawImage(img, 0, titleH, w, h)
            canvas.toBlob(resolve, 'image/png')
          }
          img.onerror = () => resolve(null)
          img.src = dataUrl
        })

        if (blob) {
          const buf = await blob.arrayBuffer()
          screenshots.push({ name: `charts/${chartNames[i] || 'chart_' + i}.png`, data: btoa(String.fromCharCode(...new Uint8Array(buf))) })
        }
      }
    }

    // Generate filament summary CSV
    if (filamentSummary.length > 0) {
      const headers = ['Filament ID', 'Host Cell ID', 'First Frame', 'Last Frame', 'Frame Count', 'Avg Length (µm)', 'Max Length (µm)', 'Avg Area (µm²)', 'Avg Eccentricity']
      const csvRows = [headers.join(',')]
      for (const f of filamentSummary) {
        csvRows.push([
          f.filament_ID, f.host_cell_ID, f.first_frame, f.last_frame,
          f.frame_count, f.avg_length, f.max_length, f.avg_area, f.avg_eccentricity
        ].join(','))
      }
      const csvStr = csvRows.join('\n')
      screenshots.push({ name: 'filament_summary.csv', data: btoa(csvStr) })
    }

    return screenshots
  }, [filamentSummary])

  const handleDownload = useCallback(async (mode) => {
    if (!currentDataset || downloading) return
    setDownloading(mode)
    try {
      // Capture screenshots before requesting zip
      const screenshots = await captureScreenshots()

      const resp = await fetch('/api/dataset/' + currentDataset.id + '/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, screenshots }),
      })
      if (!resp.ok) {
        let message = 'Build zip failed'
        try {
          const payload = await resp.json()
          if (payload?.error) message = payload.error
        } catch {
          // Ignore JSON parse failures and keep the default message.
        }
        throw new Error(message)
      }

      const blob = await resp.blob()
      const url = URL.createObjectURL(blob)
      const contentDisposition = resp.headers.get('Content-Disposition')
      const fallbackName = currentDataset.id + '_' + mode + '.zip'
      const filename = getDownloadFilename(contentDisposition, fallbackName)
      const link = document.createElement('a')

      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (e) {
      console.error('Download error:', e)
    } finally {
      setDownloading(null)
    }
  }, [currentDataset, downloading, captureScreenshots])

  if (!currentDataset) return (
    <div className="flex-1 flex items-center justify-center text-gray-500">
      <div className="text-center">
        <p className="text-lg mb-2">No dataset selected</p>
        <button onClick={onNavigateTuning} className="text-sm text-blue-400 hover:text-blue-400">← Go to Tuning</button>
      </div>
    </div>
  )

  return (
    <div className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-auto lg:overflow-hidden">
      {/* LEFT: Image + Controls */}
      <div className="w-full lg:w-[42%] flex flex-col shrink-0 lg:shrink lg:min-h-0 border-b lg:border-b-0 lg:border-r border-gray-800">
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
              style={{ imageRendering: viewMode === 'diagnostic' ? 'auto' : 'pixelated', aspectRatio: '1 / 1', objectFit: 'contain' }}
              onClick={handleFrameClick}
              onError={() => setImgError(true)}
            />
          )}
        </div>

        {/* Controls */}
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
              className={`p-1.5 rounded-md transition-colors ${loop ? 'bg-blue-700 text-white' : 'bg-gray-800 hover:bg-gray-700'}`} title="Loop">
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

      {/* RIGHT: Track Inspector + Charts */}
      <div className="flex-1 min-w-0 flex flex-col shrink-0 lg:shrink lg:min-h-0">
        {/* Run name */}
        <div className="flex-none px-4 pt-3 pb-2 border-b border-gray-800 bg-gray-900/80">
          <p className="text-sm text-gray-200 truncate px-3 py-1.5">{summary?.run_name || currentDataset.name}</p>
        </div>
        <div className="flex-none border-b border-gray-800 min-h-[250px]" style={{ height: '45%' }}>
          <TrackInspector
            filamentSummary={filamentSummary}
            rows={rows}
            selectedFilamentId={selectedFilamentId}
            onSelectFilament={setSelectedFilamentId}
            onJumpToFrame={jumpToFrame}
            currentFrame={frame}
            onFilterChange={setFilteredFilamentIds}
          />
        </div>
        <div className="flex-1 min-h-[400px] lg:min-h-0" ref={chartsRef}>
          <MetricsDashboard
            rows={rows}
            filamentSummary={filamentSummary}
            onJumpToFrame={jumpToFrame}
            filteredFilamentIds={filteredFilamentIds}
          />
        </div>
        {/* Download buttons */}
        <div className="flex-none p-3 sm:p-4 border-t border-gray-800 bg-gray-900/80 flex flex-col sm:flex-row gap-2">
          <button
            onClick={() => handleDownload('results')}
            disabled={!!downloading}
            className="flex-1 py-2.5 rounded-lg border border-gray-700 bg-gray-800 hover:bg-gray-700 text-sm font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {downloading === 'results' ? (
              <><svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg> Preparing...</>
            ) : (
              <><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg> Download Results</>
            )}
          </button>
          <button
            onClick={() => handleDownload('all')}
            disabled={!!downloading}
            className="flex-1 py-2.5 rounded-lg border border-gray-700 bg-gray-800 hover:bg-gray-700 text-sm font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {downloading === 'all' ? (
              <><svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg> Preparing...</>
            ) : (
              <><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg> Download All Files</>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
