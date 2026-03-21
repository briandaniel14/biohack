import { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine, Legend,
} from 'recharts'
import { FRAME_COUNT, getTrackColor } from './data.js'

const TT = {
  background: '#1f2937', border: 'none', fontSize: 10, padding: '4px 8px', borderRadius: 4
}

export default function MetricsDashboard({ measurements, trackSummary }) {
  const trackIds = useMemo(() => {
    return [...new Set(trackSummary.map(t => t.track_id))].sort((a, b) => a - b)
  }, [trackSummary])

  // 1. Length of all filaments over time (major_axis per track per frame) + max hlines
  const lengthData = useMemo(() => {
    const data = []
    for (let f = 0; f < FRAME_COUNT; f++) {
      const row = { frame: f }
      measurements.filter(m => m.frame === f).forEach(m => {
        row[`t${m.track_id}`] = m.major_axis
      })
      data.push(row)
    }
    return data
  }, [measurements])

  const maxLengths = useMemo(() => {
    return trackIds.map(tid => {
      const vals = measurements.filter(m => m.track_id === tid).map(m => m.major_axis)
      return { tid, max: vals.length > 0 ? Math.max(...vals) : 0 }
    }).filter(v => v.max > 0)
  }, [measurements, trackIds])

  // 2. Filaments per timepoint
  const countData = useMemo(() => {
    const data = []
    for (let f = 0; f < FRAME_COUNT; f++) {
      data.push({ frame: f, count: measurements.filter(m => m.frame === f).length })
    }
    return data
  }, [measurements])

  // 3. Eccentricity over time per track
  const eccData = useMemo(() => {
    const data = []
    for (let f = 0; f < FRAME_COUNT; f++) {
      const row = { frame: f }
      measurements.filter(m => m.frame === f).forEach(m => {
        row[`t${m.track_id}`] = m.eccentricity
      })
      data.push(row)
    }
    return data
  }, [measurements])

  // 4. Major & minor axis over time per track
  const axisData = useMemo(() => {
    const data = []
    for (let f = 0; f < FRAME_COUNT; f++) {
      const row = { frame: f }
      measurements.filter(m => m.frame === f).forEach(m => {
        row[`maj_t${m.track_id}`] = m.major_axis
        row[`min_t${m.track_id}`] = m.minor_axis
      })
      data.push(row)
    }
    return data
  }, [measurements])

  return (
    <div className="grid grid-cols-2 grid-rows-2 gap-1 p-2 h-full">
      {/* Chart 1: Length over time with max hlines */}
      <ChartPanel title="Filament Length Over Time (with max lines)">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={lengthData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="frame" tick={{ fontSize: 9, fill: '#9ca3af' }} />
            <YAxis tick={{ fontSize: 9, fill: '#9ca3af' }} label={{ value: 'px', angle: -90, position: 'insideLeft', style: { fontSize: 9, fill: '#9ca3af' } }} />
            <Tooltip contentStyle={TT} labelFormatter={v => `Frame ${v}`} />
            {trackIds.map(tid => (
              <Line key={tid} type="monotone" dataKey={`t${tid}`} stroke={getTrackColor(tid)} dot={false} strokeWidth={1.5} connectNulls={false} name={`Track ${tid}`} />
            ))}
            {maxLengths.map(({ tid, max }) => (
              <ReferenceLine key={`max-${tid}`} y={max} stroke={getTrackColor(tid)} strokeDasharray="4 2" strokeWidth={1} strokeOpacity={0.5} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>

      {/* Chart 2: Filaments per timepoint */}
      <ChartPanel title="Filaments Per Timepoint">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={countData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="frame" tick={{ fontSize: 9, fill: '#9ca3af' }} />
            <YAxis tick={{ fontSize: 9, fill: '#9ca3af' }} allowDecimals={false} />
            <Tooltip contentStyle={TT} labelFormatter={v => `Frame ${v}`} />
            <Line type="stepAfter" dataKey="count" stroke="#60a5fa" dot={false} strokeWidth={1.5} name="Count" />
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>

      {/* Chart 3: Eccentricity */}
      <ChartPanel title="Eccentricity Over Time (0 = round, 1 = elongated)">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={eccData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="frame" tick={{ fontSize: 9, fill: '#9ca3af' }} />
            <YAxis tick={{ fontSize: 9, fill: '#9ca3af' }} domain={[0, 1]} />
            <Tooltip contentStyle={TT} labelFormatter={v => `Frame ${v}`} />
            {trackIds.map(tid => (
              <Line key={tid} type="monotone" dataKey={`t${tid}`} stroke={getTrackColor(tid)} dot={false} strokeWidth={1.5} connectNulls={false} name={`Track ${tid}`} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>

      {/* Chart 4: Major and Minor axis */}
      <ChartPanel title="Major & Minor Axis Length Over Time">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={axisData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="frame" tick={{ fontSize: 9, fill: '#9ca3af' }} />
            <YAxis tick={{ fontSize: 9, fill: '#9ca3af' }} label={{ value: 'px', angle: -90, position: 'insideLeft', style: { fontSize: 9, fill: '#9ca3af' } }} />
            <Tooltip contentStyle={TT} labelFormatter={v => `Frame ${v}`} />
            {trackIds.map(tid => (
              <Line key={`maj-${tid}`} type="monotone" dataKey={`maj_t${tid}`} stroke={getTrackColor(tid)} dot={false} strokeWidth={1.5} connectNulls={false} name={`T${tid} major`} />
            ))}
            {trackIds.map(tid => (
              <Line key={`min-${tid}`} type="monotone" dataKey={`min_t${tid}`} stroke={getTrackColor(tid)} dot={false} strokeWidth={1} strokeDasharray="4 2" connectNulls={false} name={`T${tid} minor`} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>
    </div>
  )
}

function ChartPanel({ title, children }) {
  return (
    <div className="bg-gray-900 rounded p-2 flex flex-col min-h-0">
      <div className="text-[10px] text-white font-semibold uppercase mb-1 flex-none">{title}</div>
      <div className="flex-1 min-h-0">{children}</div>
    </div>
  )
}
