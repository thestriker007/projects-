import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  XOctagon,
  Thermometer,
  Zap,
  Loader,
  Cpu
} from 'lucide-react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';

// Fallback universal machines if API fails
const FALLBACK_MACHINES = [
  { id: 'UMA-01', name: 'Universal Machine Alpha', status: 'Healthy', rul_days: 145, cluster_alpha: 518.67, cluster_beta: 641.82, cluster_gamma: 1589.70 },
  { id: 'UMB-02', name: 'Universal Machine Beta', status: 'Warning', rul_days: 12, cluster_alpha: 518.67, cluster_beta: 642.15, cluster_gamma: 1591.82 },
];

// Initial Mock History to keep charts looking nice
const MOCK_HISTORY = Array.from({ length: 30 }).map((_, i) => ({
  day: i,
  rul_A: 175 - i,
  rul_B: 42 - i,
  rul_G: 32 - i,
}));

// Components
const StatusBadge = ({ status }) => {
  const iconMap = {
    Healthy: <CheckCircle size={14} style={{ marginRight: '4px' }} />,
    Warning: <AlertTriangle size={14} style={{ marginRight: '4px' }} />,
    Danger: <XOctagon size={14} style={{ marginRight: '4px' }} />
  };
  return (
    <span className={`badge ${status.toLowerCase()}`}>
      {iconMap[status]} {status}
    </span>
  );
};

export default function App() {
  const [machines, setMachines] = useState([]);
  const [selectedMachine, setSelectedMachine] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMachines = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/machines');
        if (!response.ok) throw new Error("API Offline");
        const data = await response.json();
        setMachines(data);
        setSelectedMachine(prev => {
          if (!prev) return data[0];
          return data.find(m => m.id === prev.id) || data[0];
        });
        setLoading(false);
      } catch (err) {
        console.error("Error fetching from backend, using fallback:", err);
        setMachines(FALLBACK_MACHINES);
        setSelectedMachine(prev => prev || FALLBACK_MACHINES[0]);
        setLoading(false);
      }
    };

    fetchMachines();
    const interval = setInterval(fetchMachines, 3000); // Simulate real-time streaming
    return () => clearInterval(interval);
  }, []);

  if (loading || !selectedMachine) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', padding: '2rem' }}>
         <div className="glass-panel animate-pulse" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '3rem', borderRadius: '24px' }}>
            <Loader className="animate-spin-slow" color="#3b82f6" size={64} style={{ marginBottom: '1.5rem' }} />
            <h1 className="text-gradient" style={{ fontSize: '2rem', marginBottom: '0.5rem', fontWeight: 700 }}>Initializing Universal Architectures...</h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>Establishing secure connection to telemetry matrix</p>
         </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', gap: '2rem', padding: '1.5rem', minHeight: '100vh', maxWidth: '1600px', margin: '0 auto' }}>

      {/* Sidebar Overview */}
      <div className="glass-panel" style={{ width: '350px', display: 'flex', flexDirection: 'column', gap: '1.5rem', height: 'fit-content' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Cpu color="#3b82f6" size={32} />
          <h2 className="text-gradient" style={{ fontSize: '1.5rem', margin: 0, letterSpacing: '0.05em' }}>SYNC</h2>
        </div>

        <div>
          <h4 style={{ color: 'var(--text-secondary)', marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em', fontSize: '0.75rem' }}>Fleet Status</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {machines.map((m) => (
              <div
                key={m.id}
                className="glass-panel hover-lift"
                style={{
                  padding: '1rem',
                  cursor: 'pointer',
                  border: selectedMachine.id === m.id ? '1px solid var(--accent-color)' : '',
                  background: selectedMachine.id === m.id ? 'rgba(59, 130, 246, 0.05)' : ''
                }}
                onClick={() => setSelectedMachine(m)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <h4 style={{ margin: 0, fontSize: '1.1rem' }}>{m.name}</h4>
                  <StatusBadge status={m.status} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                  <span>ID: {m.id}</span>
                  <span>RUL: <strong style={{ color: 'var(--text-primary)' }}>{m.rul_days} Days</strong></span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Dashboard */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

        {/* Header Details */}
        <div className="glass-panel" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '2rem' }}>
          <div>
            <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>{selectedMachine.name}</h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>Detailed telemetry and remaining useful life predictions</p>
          </div>
          <div style={{ transform: 'scale(1.2)' }}>
            <StatusBadge status={selectedMachine.status} />
          </div>
        </div>

        {/* Metrics Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
          <div className="glass-panel hover-lift" style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', padding: '1.5rem' }}>
             <div style={{ padding: '1rem', backgroundColor: 'rgba(59, 130, 246, 0.1)', borderRadius: '12px' }}>
                <Activity color="#3b82f6" size={28} />
             </div>
             <div>
               <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '0.25rem' }}>Diagnostic Array Alpha</p>
               <h3 style={{ margin: 0, fontSize: '1.75rem' }}>{selectedMachine.cluster_alpha} <span style={{fontSize: '1rem', color: 'var(--text-secondary)'}}>ν</span></h3>
             </div>
          </div>
          <div className="glass-panel hover-lift" style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', padding: '1.5rem' }}>
             <div style={{ padding: '1rem', backgroundColor: 'rgba(245, 158, 11, 0.1)', borderRadius: '12px' }}>
                <Activity color="#f59e0b" size={28} />
             </div>
             <div>
               <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '0.25rem' }}>Diagnostic Array Beta</p>
               <h3 style={{ margin: 0, fontSize: '1.75rem' }}>{selectedMachine.cluster_beta} <span style={{fontSize: '1rem', color: 'var(--text-secondary)'}}>η</span></h3>
             </div>
          </div>
          <div className="glass-panel hover-lift" style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', padding: '1.5rem' }}>
             <div style={{ padding: '1rem', backgroundColor: 'rgba(239, 68, 68, 0.1)', borderRadius: '12px' }}>
                <Activity color="#ef4444" size={28} />
             </div>
             <div>
               <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '0.25rem' }}>Diagnostic Array Gamma</p>
               <h3 style={{ margin: 0, fontSize: '1.75rem' }}>{selectedMachine.cluster_gamma} <span style={{fontSize: '1rem', color: 'var(--text-secondary)'}}>λ</span></h3>
             </div>
          </div>
        </div>

        {/* Chart Area */}
        <div className="glass-panel" style={{ flex: 1, minHeight: '400px', display: 'flex', flexDirection: 'column', padding: '2rem' }}>
          <h3 style={{ marginBottom: '2rem', fontSize: '1.25rem' }}>RUL Forecast (Remaining Useful Life)</h3>
          <div style={{ flex: 1, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={MOCK_HISTORY} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorRul" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--glass-border)" vertical={false} />
                <XAxis dataKey="day" stroke="var(--text-secondary)" tick={{ fill: 'var(--text-secondary)' }} axisLine={false} tickLine={false} />
                <YAxis stroke="var(--text-secondary)" tick={{ fill: 'var(--text-secondary)' }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--glass-border)', borderRadius: '8px', color: '#fff', boxShadow: '0 4px 20px rgba(0,0,0,0.5)' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Area type="monotone" dataKey={`rul_${selectedMachine.id.split('-')[1] || 'A'}`} stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorRul)" name="RUL (Days)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}
