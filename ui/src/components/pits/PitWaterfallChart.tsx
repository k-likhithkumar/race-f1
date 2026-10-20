import React from 'react';
import { TimeLossBreakdownItem } from '../../types/pits';
import { ShieldCheck, AlertCircle, Info } from 'lucide-react';

interface PitWaterfallChartProps {
  breakdown: TimeLossBreakdownItem | null;
  driver: string;
}

export const PitWaterfallChart: React.FC<PitWaterfallChartProps> = ({ breakdown, driver }) => {
  if (!breakdown || !breakdown.loss_factors_s) {
    return (
      <div className="waterfall-empty-card" id="waterfall-empty-state">
        <Info size={24} className="text-muted" />
        <div>No time-loss attribution data available for {driver} at this point in the race.</div>
      </div>
    );
  }

  const factors = breakdown.loss_factors_s;
  const totalDeficit = breakdown.total_deficit_vs_baseline_s;

  const components = [
    {
      key: 'caution',
      label: 'Caution & Incidents',
      sublabel: 'SC / VSC / Yellow Flag lap deficits',
      value: factors.caution_and_incidents_s,
      color: '#ffb800',
      description: 'Laps under confirmed caution codes (2, 4, 5, 6, 7). Unknown status is excluded.',
    },
    {
      key: 'pit',
      label: 'Pit Stop & Transit',
      sublabel: 'In-lap & out-lap total deficit',
      value: factors.pit_stop_and_transit_s,
      color: '#00d2ff',
      description: 'Encompasses pit lane traversal, wheel change, and deceleration/acceleration without double-counting.',
    },
    {
      key: 'traffic',
      label: 'Traffic & Dirty Air',
      sublabel: 'Deficit within 1.2s wake of car ahead',
      value: factors.traffic_s,
      color: '#ff8a00',
      description: 'Time lost on green non-pit laps while trailing closely behind competitor.',
    },
    {
      key: 'tyre',
      label: 'Tyre Degradation',
      sublabel: 'Pace drop on tyres older than 5 laps',
      value: factors.tyre_degradation_s,
      color: '#b347ff',
      description: 'Progressive compound wear on free-air laps based on fuel-adjusted pace falloff.',
    },
    {
      key: 'residual',
      label: 'Balancing Residual',
      sublabel: 'Driver variance, setup & energy mgmt',
      value: factors.unexplained_residual_s,
      color: '#8e8e93',
      description: 'Mathematically exact balancing residual preserving identity within ±0.01s.',
    },
  ];

  // Maximum value for proportional bar rendering
  const maxBarValue = Math.max(
    Math.abs(totalDeficit),
    ...components.map((c) => Math.abs(c.value)),
    1.0
  );

  return (
    <div className="waterfall-container" id="time-loss-waterfall-card">
      <div className="waterfall-header">
        <div>
          <div className="waterfall-title">Time-Loss Attribution Waterfall</div>
          <div className="waterfall-subtitle">
            Non-overlapping decomposition vs clean field median ({breakdown.reference_clean_lap_pace_s.toFixed(3)}s)
          </div>
        </div>

        <div className={`identity-badge ${breakdown.accounting_identity_verified ? 'verified' : 'unverified'}`}>
          {breakdown.accounting_identity_verified ? (
            <>
              <ShieldCheck size={14} /> Identity Verified (±0.01s)
            </>
          ) : (
            <>
              <AlertCircle size={14} /> Identity Discrepancy
            </>
          )}
        </div>
      </div>

      {/* Summary KPI Banner */}
      <div className="waterfall-summary-row">
        <div className="waterfall-kpi-item">
          <span className="kpi-label">Driver Total Elapsed</span>
          <span className="kpi-val font-mono">{breakdown.driver_total_time_s.toFixed(2)}s</span>
        </div>
        <div className="waterfall-kpi-item">
          <span className="kpi-label">Total Measured Deficit</span>
          <span className="kpi-val font-mono text-amber">
            {totalDeficit >= 0 ? `+${totalDeficit.toFixed(2)}` : totalDeficit.toFixed(2)}s
          </span>
        </div>
        <div className="waterfall-kpi-item">
          <span className="kpi-label">Laps Analyzed</span>
          <span className="kpi-val font-mono">Lap {breakdown.as_of_lap}</span>
        </div>
      </div>

      {/* Waterfall Bars */}
      <div className="waterfall-bars-list">
        {components.map((item) => {
          const pct = Math.min(100, (Math.abs(item.value) / maxBarValue) * 100);
          const isNegative = item.value < 0;

          return (
            <div key={item.key} className="waterfall-bar-row">
              <div className="waterfall-bar-label-group">
                <span className="bar-label">{item.label}</span>
                <span className="bar-sublabel">{item.sublabel}</span>
              </div>

              <div className="waterfall-bar-track">
                <div
                  className="waterfall-bar-fill"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: item.color,
                  }}
                />
              </div>

              <div className="waterfall-bar-val font-mono font-bold" style={{ color: item.color }}>
                {isNegative ? `${item.value.toFixed(2)}s` : `+${item.value.toFixed(2)}s`}
              </div>
            </div>
          );
        })}
      </div>

      {/* Total Accounting Identity Equation Box */}
      <div className="waterfall-equation-box">
        <div className="equation-title">Strict Accounting Identity:</div>
        <div className="equation-math font-mono">
          <span>Deficit ({totalDeficit.toFixed(2)}s)</span> ={' '}
          <span style={{ color: '#ffb800' }}>Caution ({factors.caution_and_incidents_s.toFixed(2)}s)</span> +{' '}
          <span style={{ color: '#00d2ff' }}>Pit ({factors.pit_stop_and_transit_s.toFixed(2)}s)</span> +{' '}
          <span style={{ color: '#ff8a00' }}>Traffic ({factors.traffic_s.toFixed(2)}s)</span> +{' '}
          <span style={{ color: '#b347ff' }}>Tyre ({factors.tyre_degradation_s.toFixed(2)}s)</span> +{' '}
          <span style={{ color: '#8e8e93' }}>Residual ({factors.unexplained_residual_s.toFixed(2)}s)</span>
        </div>
      </div>

      {/* Attribution Notes */}
      {breakdown.attribution_notes && breakdown.attribution_notes.length > 0 && (
        <ul className="waterfall-notes-list">
          {breakdown.attribution_notes.map((note, idx) => (
            <li key={idx}>{note}</li>
          ))}
        </ul>
      )}
    </div>
  );
};
