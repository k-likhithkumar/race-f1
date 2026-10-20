import React from 'react';
import { AssistantKeyNumber } from '../../types/assistant';

interface KeyNumberBadgeProps {
  keyNumber: AssistantKeyNumber;
}

export const KeyNumberBadge: React.FC<KeyNumberBadgeProps> = ({ keyNumber }) => {
  return (
    <div className="key-number-card">
      <span className="key-number-label">{keyNumber.label}</span>
      <span className="key-number-val mono">
        {keyNumber.value}
        {keyNumber.unit && <span className="key-number-unit">{keyNumber.unit}</span>}
      </span>
    </div>
  );
};
