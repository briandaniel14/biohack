import { useState, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { getTrackColor, frameToTimestamp } from './data.js'

const SORT_KEYS = [
  { key: 'track_id', label: 'ID' },
  { key: 'morphology_label', label: 'Morph' },
  { key: 'lifetime_frames', label: 'Lifetime' },
  { key: 'formation_frame', label: 'Start' },
  { key: 'max_length', label: 'Max Len' },
]

export default function TrackInspector({
  trackSummary,
  measurements,
  selectedTrackId,
  onSelectTrack,
  onJumpToFrame,
  currentFrame,
}) {
  const [sortKey, setSortKey] = useState('track_id')
  const [sortAsc, setSortAsc] = useState(true)

  const sortedTracks = useMemo(() => {
    const sorted = [...trackSummary].sort((a, b) => {
      const va = a[sortKey]
      const vb = b[sortKey]
      if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va)
      return sortAsc ? va - vb : vb - va
    })
    return sorted
  }, [trackSummary, sortKey, sortAsc])

  const selectedTrack = useMemo(
    () => trackSummary.find(t => t.track_id === selectedTrackId),
    [trackSummary, selectedTrackId]
  )

  const selectedTrackMeasurements = useMemo(() => {
    if (selectedTrackId == null) return []
    return measurements
      .filter(m => m.track_id === selectedTrackId)
      .sort((a, b) => a.frame - b.frame)
  }, [measurements, selectedTrackId])

  const handleSort = (key) => {
    if (key === sortKey) setSortAsc(!sortAsc)
    else { setSortKey(key); setSortAsc(true) }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-3 py-2 bg-gray-900 text-xs font-semibold text-white uppercase tracking-wide border-b border-gray-800">
        Track Inspector — {trackSummary.length} tracks
      </div>

      {/* Track table */}
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
            {sortedTracks.map(t => {
              const isActive = currentFrame >= t.formation_frame && currentFrame <= t.dissolution_frame
              return (
                <tr
                  key={t.track_id}
                  onClick={() => onSelectTrack(t.track_id)}
                  className={`cursor-pointer border-b border-gray-800/50 hover:bg-gray-800 ${
                    t.track_id === selectedTrackId ? 'bg-green-900/40' : ''
                  } ${isActive ? '' : 'opacity-50'}`}
                >
                  <td className="px-2 py-1">
                    <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: getTrackColor(t.track_id) }} />
                    {t.track_id}
                  </td>
                  <td className="px-2 py-1">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                      t.morphology_label === 'filament' ? 'bg-green-900 text-green-300' :
                      t.morphology_label === 'condensate' ? 'bg-purple-900 text-purple-300' :
                      'bg-yellow-900 text-yellow-300'
                    }`}>
                      {t.morphology_label}
                    </span>
                  </td>
                  <td className="px-2 py-1 font-mono">{t.lifetime_frames}f</td>
                  <td className="px-2 py-1 font-mono">{t.formation_frame}</td>
                  <td className="px-2 py-1 font-mono">{t.max_length?.toFixed(1)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Selected track detail */}
      {selectedTrack && (
        <div className="flex-none border-t border-gray-800 bg-gray-900/50 p-3 overflow-auto" style={{ maxHeight: 260 }}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: getTrackColor(selectedTrack.track_id) }} />
              <span className="font-semibold text-sm">Track {selectedTrack.track_id}</span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                selectedTrack.morphology_label === 'filament' ? 'bg-green-900 text-green-300' :
                selectedTrack.morphology_label === 'condensate' ? 'bg-purple-900 text-purple-300' :
                'bg-yellow-900 text-yellow-300'
              }`}>
                {selectedTrack.morphology_label}
              </span>
            </div>
            <button
              onClick={() => onJumpToFrame(selectedTrack.formation_frame)}
              className="px-2 py-0.5 bg-green-700 hover:bg-green-600 rounded text-xs"
            >
              Jump to start
            </button>
          </div>

          <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs mb-3">
            <div>
              <span className="text-gray-500">Lifetime:</span>{' '}
              <span className="font-mono">{selectedTrack.lifetime_frames}f ({selectedTrack.lifetime_minutes?.toFixed(0)}min)</span>
            </div>
            <div>
              <span className="text-gray-500">Frames:</span>{' '}
              <span className="font-mono">{selectedTrack.formation_frame} → {selectedTrack.dissolution_frame}</span>
            </div>
            <div>
              <span className="text-gray-500">Transitions:</span>{' '}
              <span className="font-mono">{selectedTrack.num_transitions}</span>
            </div>
            <div>
              <span className="text-gray-500">Max length:</span>{' '}
              <span className="font-mono">{selectedTrack.max_length?.toFixed(1)}px</span>
            </div>
            <div>
              <span className="text-gray-500">Init morph:</span>{' '}
              <span className="font-mono">{selectedTrack.initial_morphology}</span>
            </div>
            <div>
              <span className="text-gray-500">Final morph:</span>{' '}
              <span className="font-mono">{selectedTrack.final_morphology}</span>
            </div>
          </div>

          {/* Mini sparklines */}
          {selectedTrackMeasurements.length > 1 && (
            <div className="grid grid-cols-3 gap-2">
              <MiniChart data={selectedTrackMeasurements} dataKey="major_axis" label="Length" color="#60a5fa" />
              <MiniChart data={selectedTrackMeasurements} dataKey="mean_intensity" label="Intensity" color="#f59e0b" />
              <MiniChart data={selectedTrackMeasurements} dataKey="eccentricity" label="Eccentricity" color="#a78bfa" />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function MiniChart({ data, dataKey, label, color }) {
  return (
    <div>
      <div className="text-[10px] text-gray-500 mb-0.5">{label}</div>
      <ResponsiveContainer width="100%" height={50}>
        <LineChart data={data}>
          <XAxis dataKey="frame" hide />
          <YAxis hide domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{ background: '#1f2937', border: 'none', fontSize: 10, padding: '4px 8px' }}
            labelFormatter={v => `Frame ${v}`}
            formatter={v => [typeof v === 'number' ? v.toFixed(2) : v, label]}
          />
          <Line type="monotone" dataKey={dataKey} stroke={color} dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
