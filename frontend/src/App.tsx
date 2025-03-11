import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./components/theme-provider";
import { Toaster } from "./components/ui/toaster";

// Components
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";

// Pages
import Dashboard from "./pages/Dashboard";
import Streams from "./pages/Streams";
import AlertsPage from "./pages/Alerts";
import Settings from "./pages/Settings";
import Models from "./pages/Models";

/**
 * Main application component
 * Rules: Functional component, TypeScript interface, Declarative JSX
 */
interface AppProps {}

function App({}: AppProps): JSX.Element {
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);

  function toggleSidebar(): void {
    setSidebarOpen(!sidebarOpen);
  }

  return (
    <ThemeProvider defaultTheme="light" storageKey="anomaly-detection-theme">
      <div className="min-h-screen bg-background">
        <Navbar toggleSidebar={toggleSidebar} />
        <div className="flex">
          <Sidebar open={sidebarOpen} />
          <main className={`flex-1 p-4 ${sidebarOpen ? 'sm:ml-64' : 'sm:ml-0'} transition-all duration-300`}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/streams" element={<Streams />} />
              <Route path="/alerts" element={<AlertsPage />} />
              <Route path="/models" element={<Models />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>
        </div>
        <Toaster />
      </div>
    </ThemeProvider>
  );
}

export default App;
