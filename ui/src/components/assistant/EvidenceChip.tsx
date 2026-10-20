import React from 'react';
import { EvidenceReference } from '../../types/assistant';
import {
  Flag,
  ArrowRightLeft,
  Disc,
  Wrench,
  Swords,
  ExternalLink,
} from 'lucide-react';

interface EvidenceChipProps {
  evidence: EvidenceReference;
  onNavigate: (evidence: EvidenceReference) => void;
}

export const EvidenceChip: React.FC<EvidenceChipProps> = ({ evidence, onNavigate }) => {
  const getIcon = () => {
    switch (evidence.type) {
      case 'replay':
        return <Flag size={13} className="ev-icon ev-icon-replay" />;
      case 'comparison':
        return <ArrowRightLeft size={13} className="ev-icon ev-icon-comparison" />;
      case 'tyres':
        return <Disc size={13} className="ev-icon ev-icon-tyres" />;
      case 'pits':
        return <Wrench size={13} className="ev-icon ev-icon-pits" />;
      case 'battles':
        return <Swords size={13} className="ev-icon ev-icon-battles" />;
      default:
        return <ExternalLink size={13} className="ev-icon" />;
    }
  };

  const getTargetLabel = () => {
    switch (evidence.type) {
      case 'replay':
        return 'Replay';
      case 'comparison':
        return 'Comparison';
      case 'tyres':
        return 'Tyres';
      case 'pits':
        return 'Pits';
      case 'battles':
        return 'Battles';
      default:
        return 'View';
    }
  };

  return (
    <button
      type="button"
      className={`evidence-chip evidence-chip-${evidence.type}`}
      id={`evidence-chip-${evidence.id}`}
      onClick={() => onNavigate(evidence)}
      title={`${evidence.description} Click to navigate to ${getTargetLabel()} view.`}
      aria-label={`Evidence: ${evidence.label}. Click to jump to ${getTargetLabel()} view.`}
    >
      <span className="evidence-chip-icon-wrap">{getIcon()}</span>
      <span className="evidence-chip-label">{evidence.label}</span>
      <span className="evidence-chip-badge">{getTargetLabel()}</span>
      <ExternalLink size={11} className="evidence-chip-jump-icon" />
    </button>
  );
};
