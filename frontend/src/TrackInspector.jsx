import { useState, useMemo } from 'react'
import { getTrackColor, frameToTimestamp } from './data.js'

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
}) {
  const [sortKey, setSortKey] = useState('filament_ID')
  const [sortAsc, setSortAsc] = useState(true)

  const sortedFilaments = useMemo(() => {
    return [...filamentSummary].sort((a, b) => {
      const va = a[sortKey] ?? 0
      const vb = b[sortKey] ?? 0
      return sortAsc ? va - vb : vb - va
    })
  }, [filamentSummary, sortKey, sortAsc])

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
      <div className="px-3 py-2 bg-gray-900 text-xs font-semibold text-white uppercase tracking-wide border-b border-gray-800">
        Filament Tracker — {filamentSummary.length} filaments
      </div>

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
                  <td className="px-2 py-1 font-mono">{f.avg_length}px</td>
                  <td className="px-2 py-1 font-mono">f{Math.round(f.time_of_appearance || f.first_frame)}</td>
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

          <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs mb-3">
            <div>
              <span className="text-gray-500">Frames:</span>{' '}
              <span className="font-mono">{selectedFilament.first_frame} → {selectedFilament.last_frame}</span>
            </div>
            <div>
              <span className="text-gray-500">Lifetime:</span>{' '}
              <span className="font-mono">{selectedFilament.frame_count} frames</span>
            </div>
            <div>
              <span className="text-gray-500">Appeared:</span>{' '}
              <span className="font-mono">{frameToTimestamp(selectedFilament.time_of_appearance || selectedFilament.first_frame)}</span>
            </div>
            <div>
              <span className="text-gray-500">Avg length:</span>{' '}
              <span className="font-mono">{selectedFilament.avg_length}px</span>
            </div>
            <div>
              <span className="text-gray-500">Max length:</span>{' '}
              <span className="font-mono">{selectedFilament.max_length}px</span>
            </div>
            <div>
              <span className="text-gray-500">Avg area:</span>{' '}
              <span className="font-mono">{selectedFilament.avg_area}px²</span>
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
                  f{r.frame} — {Number(r.filament_mean_length_px).toFixed(1)}px
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
