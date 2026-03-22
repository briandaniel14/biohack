import { useState, useMemo } from 'react'
import { getTrackColor, frameToTimestamp, PX_TO_UM } from './data.js'

const SORT_KEYS = [
  { key: 'filament_ID', label: 'Fil ID' },
  { key: 'host_cell_ID', label: 'Cell' },
  { key: 'frame_count', label: 'Frames' },
  { key: 'avg_length', label: 'Avg Len' },
  { key: 'time_of_appearance', label: 'Appears' },
]

export default function TrackInspector({
  filamentSummary,
  rows,
  selectedFilamentId,
  onSelectFilament,
  onJumpToFrame,
  currentFrame,
  onFilterChange,
}) {
  const [sortKey, setSortKey] = useState('filament_ID')
  const [sortAsc, setSortAsc] = useState(true)
  const [filterKey, setFilterKey] = useState('')
  const [filterOp, setFilterOp] = useState('>')
  const [filterVal, setFilterVal] = useState('')
  const [showFilter, setShowFilter] = useState(false)

  const filteredFilaments = useMemo(() => {
    if (!filterKey || filterVal === '') return filamentSummary
    const v = parseFloat(filterVal)
    if (isNaN(v)) return filamentSummary
    return filamentSummary.filter(f => {
      const fv = f[filterKey] ?? 0
      if (filterOp === '>') return fv > v
      if (filterOp === '<') return fv < v
      if (filterOp === '>=') return fv >= v
      if (filterOp === '<=') return fv <= v
      if (filterOp === '=') return Math.abs(fv - v) < 0.01
      return true
    })
  }, [filamentSummary, filterKey, filterOp, filterVal])

  // Notify parent of filtered IDs so charts can sync
  useMemo(() => {
    if (!filterKey || filterVal === '') {
      onFilterChange?.(null) // null = no filter active
    } else {
      onFilterChange?.(filteredFilaments.map(f => f.filament_ID))
    }
  }, [filteredFilaments, filterKey, filterVal, onFilterChange])

  const sortedFilaments = useMemo(() => {
    return [...filteredFilaments].sort((a, b) => {
      const va = a[sortKey] ?? 0
      const vb = b[sortKey] ?? 0
      return sortAsc ? va - vb : vb - va
    })
  }, [filteredFilaments, sortKey, sortAsc])

  const selectedFilament = useMemo(
    () => filamentSummary.find(f => f.filament_ID === selectedFilamentId),
    [filamentSummary, selectedFilamentId]
  )

  const selectedFilRows = useMemo(() => {
    if (selectedFilamentId == null) return []
    return rows
      .filter(r => r.filament_ID === selectedFilamentId && r.filament_present === 1)
      .sort((a, b) => a.frame - b.frame)
  }, [rows, selectedFilamentId])

  const handleSort = (key) => {
    if (key === sortKey) setSortAsc(!sortAsc)
    else { setSortKey(key); setSortAsc(true) }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 bg-gray-900 border-b border-gray-800 flex items-center justify-between">
        <span className="text-xs font-semibold text-white uppercase tracking-wide">
          Filament Tracker — {filteredFilaments.length}{filteredFilaments.length !== filamentSummary.length ? ` / ${filamentSummary.length}` : ''} filaments
        </span>
        <button
          onClick={() => setShowFilter(!showFilter)}
          className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
            showFilter || filterKey ? 'bg-blue-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          <svg className="w-3 h-3 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" /></svg>
          Filter
        </button>
      </div>

      {showFilter && (
        <div className="px-3 py-2 bg-gray-900/80 border-b border-gray-800 flex items-center gap-2 flex-wrap">
          <select
            value={filterKey}
            onChange={e => setFilterKey(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300 focus:outline-none focus:border-blue-500"
          >
            <option value="">Field...</option>
            <option value="avg_length">Avg Length (µm)</option>
            <option value="max_length">Max Length (µm)</option>
            <option value="avg_area">Avg Area (µm²)</option>
            <option value="frame_count">Frame Count</option>
            <option value="avg_eccentricity">Eccentricity</option>
            <option value="filament_ID">Filament ID</option>
            <option value="host_cell_ID">Cell ID</option>
          </select>
          <select
            value={filterOp}
            onChange={e => setFilterOp(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300 focus:outline-none focus:border-blue-500 w-14"
          >
            <option value=">">&gt;</option>
            <option value=">=">&ge;</option>
            <option value="<">&lt;</option>
            <option value="<=">&le;</option>
            <option value="=">=</option>
          </select>
          <input
            type="number"
            value={filterVal}
            onChange={e => setFilterVal(e.target.value)}
            placeholder="Value"
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300 w-20 font-mono focus:outline-none focus:border-blue-500"
          />
          {filterKey && filterVal !== '' && (
            <button
              onClick={() => { setFilterKey(''); setFilterVal('') }}
              className="px-2 py-1 rounded text-[11px] bg-gray-800 text-gray-400 hover:text-red-400 hover:bg-red-500/10 transition-colors"
            >Clear</button>
          )}
        </div>
      )}

      {/* Filament table */}
      <div className="flex-1 overflow-auto min-h-0">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-gray-900 z-10">
            <tr>
              {SORT_KEYS.map(sk => (
                <th
                  key={sk.key}
                  onClick={() => handleSort(sk.key)}
                  className="px-2 py-1.5 text-left cursor-pointer hover:text-white text-gray-400 select-none"
                >
                  {sk.label} {sortKey === sk.key ? (sortAsc ? '▲' : '▼') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedFilaments.map(f => {
              const isActive = currentFrame >= f.first_frame && currentFrame <= f.last_frame
              return (
                <tr
                  key={f.filament_ID}
                  onClick={() => onSelectFilament(f.filament_ID)}
                  className={`cursor-pointer border-b border-gray-800/50 hover:bg-gray-800 ${
                    f.filament_ID === selectedFilamentId ? 'bg-blue-900/40' : ''
                  } ${isActive ? '' : 'opacity-50'}`}
                >
                  <td className="px-2 py-1">
                    <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: getTrackColor(f.filament_ID) }} />
                    {Math.round(f.filament_ID)}
                  </td>
                  <td className="px-2 py-1 font-mono text-gray-400">{f.host_cell_ID}</td>
                  <td className="px-2 py-1 font-mono">{f.frame_count}</td>
                  <td className="px-2 py-1 font-mono">{f.avg_length}µm</td>
                  <td className="px-2 py-1 font-mono">f{Math.round(f.time_of_appearance || f.first_frame) + 1}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Selected filament detail */}
      {selectedFilament && (
        <div className="flex-none border-t border-gray-800 bg-gray-900/50 p-3 overflow-auto" style={{ maxHeight: 260 }}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: getTrackColor(selectedFilament.filament_ID) }} />
              <span className="font-semibold text-sm">Filament {Math.round(selectedFilament.filament_ID)}</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-blue-900 text-blue-300">
                in Cell {selectedFilament.host_cell_ID}
              </span>
            </div>
            <button
              onClick={() => onJumpToFrame(selectedFilament.first_frame)}
              className="px-2 py-0.5 bg-blue-700 hover:bg-blue-600 rounded text-xs"
            >
              Jump to start
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-4 gap-y-1 text-xs mb-3">
            <div>
              <span className="text-gray-500">Frames:</span>{' '}
              <span className="font-mono">{selectedFilament.first_frame + 1} → {selectedFilament.last_frame + 1}</span>
            </div>
            <div>
              <span className="text-gray-500">Lifetime:</span>{' '}
              <span className="font-mono">{selectedFilament.frame_count} frames</span>
            </div>

            <div>
              <span className="text-gray-500">Avg length:</span>{' '}
              <span className="font-mono">{selectedFilament.avg_length}µm</span>
            </div>
            <div>
              <span className="text-gray-500">Max length:</span>{' '}
              <span className="font-mono">{selectedFilament.max_length}µm</span>
            </div>
            <div>
              <span className="text-gray-500">Avg area:</span>{' '}
              <span className="font-mono">{selectedFilament.avg_area}µm²</span>
            </div>
            <div>
              <span className="text-gray-500">Eccentricity:</span>{' '}
              <span className="font-mono">{selectedFilament.avg_eccentricity}</span>
            </div>
          </div>

          {/* Frame-by-frame appearances */}
          <div className="mt-2">
            <div className="text-[10px] text-gray-400 uppercase tracking-wide mb-1">Frame Appearances</div>
            <div className="flex flex-wrap gap-1">
              {selectedFilRows.map(r => (
                <button
                  key={r.frame}
                  onClick={() => onJumpToFrame(r.frame)}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono transition-colors ${
                    r.frame === currentFrame
                      ? 'bg-blue-700 text-white'
                      : 'bg-blue-900/60 hover:bg-blue-800 text-blue-300'
                  }`}
                >
                  f{r.frame + 1} — {(Number(r.filament_major_axis_length) * PX_TO_UM).toFixed(2)}µm
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
