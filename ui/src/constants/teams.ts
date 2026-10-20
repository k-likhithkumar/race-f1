export const TEAM_COLORS: Record<string, string> = {
  'Red Bull Racing': '#1E41FF',
  'Ferrari': '#DC0000',
  'Mercedes': '#00D2BE',
  'McLaren': '#FF8700',
  'Aston Martin': '#006F62',
  'Alpine': '#0090FF',
  'Williams': '#005AFF',
  'AlphaTauri': '#5E8FAA',
  'RB': '#6692FF',
  'Alfa Romeo': '#900000',
  'Haas F1 Team': '#B6BABD',
  'Haas': '#B6BABD',
};

export const getTeamColor = (teamName: string | undefined): string => {
  if (!teamName) return '#8c96a8';
  return TEAM_COLORS[teamName] || '#8c96a8';
};
