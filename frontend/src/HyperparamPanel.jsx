import { useState, useRef, useCallback, useEffect } from 'react'
import { createPortal } from 'react-dom'

const PARAMS = [
  { key: 'clip_low_percentile', label: 'Clip Low Percentile', min: 0, max: 10, step: 0.1, default: 0.0,
    desc: 'Lower intensity percentile to clip before normalization.' },
  { key: 'clip_high_percentile', label: 'Clip High Percentile', min: 90, max: 100, step: 0.1, default: 100.0,
    desc: 'Upper intensity percentile for normalization.' },
  { key: 'gaussian_sigma', label: 'Gaussian Sigma', min: 0.5, max: 10, step: 0.1, default: 2.5,
    desc: 'Sigma for Gaussian blur applied to denoise.' },
  { key: 'foreground_percentile', label: 'Foreground Percentile', min: 90, max: 100, step: 0.1, default: 99.5,
    desc: 'Percentile above which pixels are foreground.' },
  { key: 'local_block_size', label: 'Local Block Size', min: 3, max: 101, step: 2, default: 35,
    desc: 'Window size for adaptive thresholding (must be odd).' },
  { key: 'local_offset', label: 'Local Offset', min: -0.1, max: 0.1, step: 0.005, default: -0.01,
    desc: 'Offset subtracted from the local threshold.' },
  { key: 'frangi_threshold_percentile', label: 'Frangi Threshold', min: 90, max: 100, step: 0.1, default: 99.5,
    desc: 'Percentile threshold for Frangi filter output.' },
  { key: 'min_object_size', label: 'Min Object Size', min: 1, max: 200, step: 1, default: 25,
    desc: 'Minimum pixels to retain a detected object.' },
  { key: 'min_pixels_for_presence', label: 'Min Pixels Presence', min: 1, max: 200, step: 1, default: 20,
    desc: 'Minimum pixels to call a filament "present."' },
]

export default function HyperparamPanel({ onChange }) {
  const [values, setValues] = useState(() =>
    Object.fromEntries(PARAMS.map(p => [p.key, p.default]))
  )
  const [tooltip, setTooltip] = useState(null)

  useEffect(() => {
    if (onChange) onChange(values)
  }, [values])

  const handleChange = (key, val) => {
    setValues(prev => ({ ...prev, [key]: val }))
  }

  const handleReset = () => {
    setValues(Object.fromEntries(PARAMS.map(p => [p.key, p.default])))
  }

  const showTooltip = useCallback((e, desc) => {
    const rect = e.currentTarget.getBoundingClientRect()
    setTooltip({ desc, x: rect.right + 8, y: rect.top })
  }, [])

  const hideTooltip = useCallback(() => setTooltip(null), [])

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 bg-gray-900/80 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-100">Hyperparameters</h3>
          <p className="text-[10px] text-gray-500 mt-0.5">Detection pipeline configuration</p>
        </div>
        <button onClick={handleReset} className="text-[11px] text-gray-500 hover:text-gray-300 px-2 py-1 rounded-md hover:bg-gray-800 transition-colors">
          Reset
        </button>
      </div>
      <div className="flex-1 overflow-auto min-h-0 p-4 space-y-2">
        {PARAMS.map(p => (
          <div key={p.key} className="rounded-lg bg-gray-800/40 px-3 py-2 flex items-center gap-3">
            <label className="w-[170px] shrink-0 text-xs text-gray-300 font-medium flex items-center gap-1 whitespace-nowrap">
              {p.label}
              <span
                className="text-gray-500 hover:text-blue-400 cursor-help text-[10px] transition-colors"
                onMouseEnter={e => showTooltip(e, p.desc)}
                onMouseLeave={hideTooltip}
              >ⓘ</span>
            </label>
            <input
              type="range"
              min={p.min}
              max={p.max}
              step={p.step}
              value={values[p.key]}
              onChange={e => handleChange(p.key, parseFloat(e.target.value))}
              className="flex-1 min-w-0 accent-blue-500 h-1.5"
            />
            <input
              type="number"
              min={p.min}
              max={p.max}
              step={p.step}
              value={values[p.key]}
              onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) handleChange(p.key, v) }}
              className="w-20 bg-gray-900 border border-gray-700 rounded-md px-2 py-0.5 text-xs font-mono text-gray-300 text-right focus:border-blue-500 focus:outline-none transition-colors shrink-0"
            />
          </div>
        ))}
      </div>
      {tooltip && createPortal(
        <div
          className="fixed z-50 w-60 p-3 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 leading-relaxed shadow-xl pointer-events-none"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          {tooltip.desc}
        </div>,
        document.body
      )}
    </div>
  )
}
