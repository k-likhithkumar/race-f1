import { PitStopItem } from '../../types/pits';
import { Clock, CheckCircle, ArrowRight } from 'lucide-react';

interface PitStopsTableProps {
  stops: PitStopItem[];
  referenceCleanPace?: number | null;
}

export const PitStopsTable: React.FC<PitStopsTableProps> = ({ stops, referenceCleanPace }) => {
  if (!stops || stops.length === 0) {
    return (
      <div className="pit-empty-card" id="pit-empty-state">
        <Clock size={28} className="text-muted" />
        <div className="pit-empty-title">No Pit Stops Recorded</div>
        <div className="pit-empty-desc">
          No pit stops have occurred on or before the current replay timestamp.
        </div>
      </div>
    );
  }

  return (
    <div className="pit-table-container" id="pit-stops-table-container">
      <div className="pit-table-header">
        <span className="pit-table-title">Pit Stop Timing & Transit Accounting</span>
        <span className="pit-table-subtitle">
          Measured loop durations vs illustrative 2.5s wheel-change assumption
        </span>
      </div>

      <div className="table-responsive">
        <table className="pit-table" id="pit-stops-table">
          <thead>
            <tr>
              <th>Stop #</th>
              <th>Status</th>
              <th>In Lap</th>
              <th>Out Lap</th>
              <th>In-Lap Time</th>
              <th>Out-Lap Time</th>
              <th>Pit Lane Duration <span className="th-tag measured">Measured</span></th>
              <th>Stationary Time <span className="th-tag assumption">Assumption</span></th>
              <th>Transit Est. <span className="th-tag estimated">Estimated</span></th>
              <th>Net Pit Loss <span className="th-tag estimated">Estimated</span></th>
              <th>Tactical Classification</th>
            </tr>
          </thead>
          <tbody>
            {stops.map((stop) => {
              const isCompleted = stop.status === 'COMPLETED';
              const isInProgress = stop.status === 'IN_PROGRESS';
              const isOutLapInProgress = stop.status === 'OUT_LAP_IN_PROGRESS';

              return (
                <tr key={stop.stop_number} id={`pit-stop-row-${stop.stop_number}`} className={`pit-row-${stop.status.toLowerCase()}`}>
                  <td className="font-mono font-bold">#{stop.stop_number}</td>

                  {/* Lifecycle Status Badge */}
                  <td>
                    {isCompleted && (
                      <span className="pit-status-chip completed">
                        <CheckCircle size={12} /> COMPLETED
                      </span>
                    )}
                    {isInProgress && (
                      <span className="pit-status-chip in-progress animate-pulse">
                        <Clock size={12} /> IN PIT LANE
                      </span>
                    )}
                    {isOutLapInProgress && (
                      <span className="pit-status-chip out-lap animate-pulse">
                        <ArrowRight size={12} /> ON OUT-LAP
                      </span>
                    )}
                  </td>

                  <td className="font-mono">L{stop.pit_in_lap}</td>
                  <td className="font-mono">L{stop.out_lap}</td>

                  {/* In-Lap Time */}
                  <td className="font-mono">
                    {stop.in_lap_time_s !== null && stop.in_lap_time_s !== undefined
                      ? `${stop.in_lap_time_s.toFixed(3)}s`
                      : '—'}
                  </td>

                  {/* Out-Lap Time */}
                  <td className="font-mono">
                    {stop.out_lap_time_s !== null && stop.out_lap_time_s !== undefined ? (
                      `${stop.out_lap_time_s.toFixed(3)}s`
                    ) : (
                      <span className="text-muted italic">In Progress</span>
                    )}
                  </td>

                  {/* Pit Lane Duration (Measured by transponder loops) */}
                  <td className="font-mono font-bold">
                    {stop.measured_pit_lane_duration_s !== null && stop.measured_pit_lane_duration_s !== undefined ? (
                      <span className="text-cyan">{stop.measured_pit_lane_duration_s.toFixed(2)}s</span>
                    ) : (
                      <span className="text-muted italic">Timing Active</span>
                    )}
                  </td>

                  {/* Stationary Wheel Change (Illustrative 2.5s Benchmark Assumption) */}
                  <td className="stationary-col">
                    <div className="stationary-cell">
                      <span className="assumption-badge">
                        {stop.reference_stationary_s.toFixed(1)}s
                      </span>
                      <span className="assumption-note">Illustrative</span>
                    </div>
                  </td>

                  {/* Estimated Transit Time (Pit Lane Duration - 2.5s Assumption) */}
                  <td className="font-mono">
                    {stop.estimated_pit_lane_transit_s !== null && stop.estimated_pit_lane_transit_s !== undefined ? (
                      `${stop.estimated_pit_lane_transit_s.toFixed(2)}s`
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>

                  {/* Estimated Net Pit Loss vs 2x Clean Lap Pace */}
                  <td className="font-mono font-bold">
                    {stop.estimated_total_pit_loss_s !== null && stop.estimated_total_pit_loss_s !== undefined ? (
                      <span className="text-amber">+{stop.estimated_total_pit_loss_s.toFixed(2)}s</span>
                    ) : (
                      <span className="text-muted italic">Pending Out-Lap</span>
                    )}
                  </td>

                  {/* Tactical Verdict Badge */}
                  <td>
                    <TacticalChip verdict={stop.tactical_verdict} isSafetyCar={stop.is_safety_car_stop} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {referenceCleanPace && (
        <div className="pit-table-footer">
          * Net pit loss calculated relative to 2 flying laps at stated clean field median ({referenceCleanPace.toFixed(3)}s).
          Pit lane duration is hardware transponder-measured; stationary time is an illustrative 2.5s benchmark.
        </div>
      )}
    </div>
  );
};

const TacticalChip: React.FC<{ verdict: string; isSafetyCar: boolean }> = ({ verdict, isSafetyCar }) => {
  if (isSafetyCar || verdict === 'SAFETY_CAR_ADVANTAGE') {
    return <span className="tactical-chip sc">SAFETY CAR</span>;
  }
  if (verdict === 'UNDERCUT_ATTEMPT') {
    return <span className="tactical-chip undercut">UNDERCUT</span>;
  }
  if (verdict === 'OVERCUT_ATTEMPT') {
    return <span className="tactical-chip overcut">OVERCUT</span>;
  }
  if (verdict === 'REGULAR_STOP') {
    return <span className="tactical-chip regular">REGULAR</span>;
  }
  return <span className="tactical-chip unknown">UNKNOWN</span>;
};
