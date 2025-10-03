
import { inject } from '@vercel/analytics';

if (import.meta.env.PROD) {
 inject();
}
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(<App />);
