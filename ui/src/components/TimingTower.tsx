import React from 'react';
import { DriverState } from '../types/replay';
import { Swords } from 'lucide-react';
import { TEAM_COLORS } from '../constants/teams';

interface TimingTowerProps {
  leaderboard: DriverState[];
  activeBattles?: any[];
}


export const TimingTower: React.FC<TimingTowerProps> = ({ leaderboard, activeBattles = [] }) => {
  const formatTime = (timeS: number | null) => {
    if (timeS === null || timeS === undefined || isNaN(timeS)) return '-';
    const m = Math.floor(timeS / 60);
    const s = (timeS % 60).toFixed(3);
    return `${m}:${s.padStart(6, '0')}`;
  };

  const formatGap = (gapS: number | null, isLeader: boolean) => {
    if (isLeader) return <span style={{ color: 'var(--f1-gold)', fontWeight: 700 }}>LEADER</span>;
    if (gapS === null || gapS === undefined || isNaN(gapS)) return '-';
    return `+${gapS.toFixed(3)}s`;
  };

  const formatInterval = (intS: number | null, isLeader: boolean) => {
    if (isLeader) return '-';
    if (intS === null || intS === undefined || isNaN(intS)) return '-';
    return `+${intS.toFixed(3)}s`;
  };

  const getTyreBadgeClass = (compound: string) => {
    const c = (compound || '').toUpperCase();
    if (c.includes('SOFT')) return 'tyre-soft';
    if (c.includes('MEDIUM')) return 'tyre-medium';
    if (c.includes('HARD')) return 'tyre-hard';
    if (c.includes('INTER')) return 'tyre-intermediate';
    if (c.includes('WET')) return 'tyre-wet';
    return 'tyre-unknown';
  };

  const getTyreLetter = (compound: string) => {
    const c = (compound || '').toUpperCase();
    if (c.includes('SOFT')) return 'S';
    if (c.includes('MEDIUM')) return 'M';
    if (c.includes('HARD')) return 'H';
    if (c.includes('INTER')) return 'I';
    if (c.includes('WET')) return 'W';
    return '?';
  };

  const getStatusBadge = (driver: DriverState) => {
    if (driver.status === 'RETIRED') {
      return <span className="status-badge badge-retired">RETIRED</span>;
    }
    if (driver.status === 'STOPPED') {
      return <span className="status-badge badge-stopped">STOPPED</span>;
    }
    if (driver.pit_status === 'IN_PIT' || driver.is_in_lap || driver.is_out_lap) {
      return <span className="status-badge badge-in-pit">PIT</span>;
    }
    if (driver.status === 'OFF_TRACK') {
      return <span className="status-badge badge-off-track">OFF-TRACK</span>;
    }
    if (driver.compound_observation_status === 'OBSERVATION_PENDING') {
      return <span className="status-badge badge-obs-pending">OBS PENDING</span>;
    }
    return <span className="status-badge badge-on-track">ON TRACK</span>;
  };

  // Find drivers in active battle
  const battleDriverSet = new Set<string>();
  for (const b of activeBattles) {
    if (b.chaser) battleDriverSet.add(b.chaser);
    if (b.defender) battleDriverSet.add(b.defender);
  }

  // Backend provides authoritative pre-sorted leaderboard
  const sorted = leaderboard;

  return (
    <div className="timing-tower-card" id="timing-tower-container">
      <div className="tower-header">
        <h2>Live Timing Tower</h2>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          {leaderboard.length} Cars Tracked
        </span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="timing-table" id="timing-tower-table">
          <thead>
            <tr>
              <th className="pos-col">Pos</th>
              <th>Driver</th>
              <th>Team</th>
              <th>Gap To Leader</th>
              <th>Interval</th>
              <th>Last Lap</th>
              <th>Tyre</th>
              <th>Stint</th>
              <th>Status</th>
              <th>Battle</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((d, index) => {
              const posNumber = index + 1;
              const isLeader = posNumber === 1;
              const isRetired = d.status === 'RETIRED';
              const teamColor = TEAM_COLORS[d.team] || '#555';
              const inBattle = battleDriverSet.has(d.driver);

              return (
                <tr
                  key={d.driver}
                  id={`driver-row-${d.driver}`}
                  className={isRetired ? 'retired-row' : ''}
                >
                  <td className={`pos-col mono ${posNumber <= 3 ? `pos-${posNumber}` : ''}`}>
                    {posNumber}
                  </td>
                  <td>
                    <div className="driver-cell">
                      <div
                        className="team-indicator"
                        style={{ backgroundColor: teamColor }}
                      />
                      <span className="driver-code">{d.driver}</span>
                    </div>
                  </td>
                  <td className="team-name">{d.team}</td>
                  <td className="mono">
                    {isRetired ? '-' : formatGap(d.timing_gap_to_leader_s, isLeader)}
                  </td>
                  <td className="mono">
                    {isRetired ? '-' : formatInterval(d.timing_interval_s, isLeader)}
                  </td>
                  <td className="mono">
                    {formatTime(d.last_lap_time_s)}
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <span
                        className={`tyre-badge ${getTyreBadgeClass(d.compound)}`}
                        title={`Compound: ${d.compound}`}
                      >
                        {getTyreLetter(d.compound)}
                      </span>
                      <span className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                        L{d.completed_tyre_age_laps}
                      </span>
                    </div>
                  </td>
                  <td className="mono" style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                    S{d.stint}
                  </td>
                  <td>
                    {getStatusBadge(d)}
                  </td>
                  <td>
                    {inBattle && (
                      <span className="battle-pill">
                        <Swords size={12} /> BATTLE
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
