import { JSX } from "react";

/**
 * Settings page component for managing system configuration
 * Rule: Functional component, TypeScript interface
 */
interface SettingsProps {}

function Settings({}: SettingsProps): JSX.Element {
  return (
    <div className="container mx-auto p-4">
      <h1 className="mb-6 text-3xl font-bold">Settings</h1>
      
      <div className="grid gap-6 md:grid-cols-2">
        <div className="rounded-lg border p-4 shadow-sm">
          <h2 className="mb-4 text-xl font-semibold">System Settings</h2>
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm font-medium">Alert Threshold</label>
              <input 
                type="range" 
                min="0" 
                max="100" 
                defaultValue="75"
                className="w-full" 
              />
              <div className="mt-2 flex justify-between text-sm">
                <span>Low</span>
                <span>Medium</span>
                <span>High</span>
              </div>
            </div>
            
            <div>
              <label className="mb-2 block text-sm font-medium">Data Retention (days)</label>
              <select className="w-full rounded-md border p-2">
                <option value="7">7 days</option>
                <option value="14">14 days</option>
                <option value="30">30 days</option>
                <option value="90">90 days</option>
                <option value="180">180 days</option>
                <option value="365">365 days</option>
              </select>
            </div>
            
            <div>
              <label className="mb-2 block text-sm font-medium">Update Frequency</label>
              <select className="w-full rounded-md border p-2">
                <option value="5">5 seconds</option>
                <option value="10">10 seconds</option>
                <option value="30">30 seconds</option>
                <option value="60">1 minute</option>
                <option value="300">5 minutes</option>
              </select>
            </div>
          </div>
        </div>
        
        <div className="rounded-lg border p-4 shadow-sm">
          <h2 className="mb-4 text-xl font-semibold">Notification Settings</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Email Alerts</span>
              <div className="relative h-6 w-12 cursor-pointer rounded-full bg-gray-200">
                <div className="absolute bottom-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition"></div>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">SMS Alerts</span>
              <div className="relative h-6 w-12 cursor-pointer rounded-full bg-gray-200">
                <div className="absolute bottom-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition"></div>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Slack Notifications</span>
              <div className="relative h-6 w-12 cursor-pointer rounded-full bg-primary">
                <div className="absolute bottom-0.5 right-0.5 h-5 w-5 rounded-full bg-white transition"></div>
              </div>
            </div>
            
            <div>
              <label className="mb-2 block text-sm font-medium">Alert Recipients</label>
              <input 
                type="text"
                placeholder="email@example.com, another@example.com"
                className="w-full rounded-md border p-2" 
              />
            </div>
          </div>
        </div>
        
        <div className="rounded-lg border p-4 shadow-sm">
          <h2 className="mb-4 text-xl font-semibold">API Integration</h2>
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm font-medium">API Key</label>
              <div className="flex">
                <input 
                  type="password"
                  value="********************************"
                  readOnly
                  className="flex-1 rounded-l-md border p-2" 
                />
                <button className="rounded-r-md bg-primary px-4 text-primary-foreground">
                  Generate
                </button>
              </div>
            </div>
            
            <div>
              <label className="mb-2 block text-sm font-medium">Webhook URL</label>
              <input 
                type="text"
                placeholder="https://your-service.com/webhook"
                className="w-full rounded-md border p-2" 
              />
            </div>
          </div>
        </div>
        
        <div className="rounded-lg border p-4 shadow-sm">
          <h2 className="mb-4 text-xl font-semibold">Advanced Settings</h2>
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm font-medium">Log Level</label>
              <select className="w-full rounded-md border p-2">
                <option value="error">Error</option>
                <option value="warn">Warning</option>
                <option value="info">Info</option>
                <option value="debug">Debug</option>
                <option value="trace">Trace</option>
              </select>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Developer Mode</span>
              <div className="relative h-6 w-12 cursor-pointer rounded-full bg-gray-200">
                <div className="absolute bottom-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition"></div>
              </div>
            </div>
            
            <div>
              <button className="mt-4 w-full rounded-md bg-destructive p-2 text-destructive-foreground">
                Reset All Settings
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-6 flex justify-end space-x-4">
        <button className="rounded-md border bg-transparent px-4 py-2">
          Cancel
        </button>
        <button className="rounded-md bg-primary px-4 py-2 text-primary-foreground">
          Save Changes
        </button>
      </div>
    </div>
  );
}

export default Settings;
