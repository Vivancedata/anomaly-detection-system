import { useState } from "react";

/**
 * Models page component for managing anomaly detection models
 * Rule: Functional component, TypeScript interface
 */
interface ModelsProps {}

interface Model {
  id: string;
  name: string;
  type: string;
  status: "active" | "inactive" | "training";
  accuracy: number;
  lastUpdated: string;
  description: string;
}

function Models({}: ModelsProps): JSX.Element {
  const [models, setModels] = useState<Model[]>([
    {
      id: "model-1",
      name: "Statistical Anomaly Detector",
      type: "Statistical",
      status: "active",
      accuracy: 94.2,
      lastUpdated: "2025-03-10",
      description: "Z-score analysis with exponential smoothing for time series data",
    },
    {
      id: "model-2",
      name: "Isolation Forest",
      type: "Machine Learning",
      status: "active",
      accuracy: 96.8,
      lastUpdated: "2025-03-08",
      description: "Isolation forest for detecting outliers in high-dimensional data",
    },
    {
      id: "model-3",
      name: "LSTM Sequence Anomaly",
      type: "Deep Learning",
      status: "active",
      accuracy: 97.5,
      lastUpdated: "2025-03-05",
      description: "Long short-term memory network for sequence anomaly detection",
    },
    {
      id: "model-4",
      name: "Autoencoder",
      type: "Deep Learning",
      status: "inactive",
      accuracy: 95.1,
      lastUpdated: "2025-02-28",
      description: "Neural network autoencoder for reconstructing normal patterns",
    },
    {
      id: "model-5",
      name: "Ensemble Model",
      type: "Ensemble",
      status: "training",
      accuracy: 0,
      lastUpdated: "2025-03-11",
      description: "Weighted ensemble of multiple anomaly detection algorithms",
    },
  ]);

  function getStatusBadgeClass(status: string): string {
    switch (status) {
      case "active":
        return "bg-green-100 text-green-800";
      case "inactive":
        return "bg-gray-100 text-gray-800";
      case "training":
        return "bg-blue-100 text-blue-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  }

  function handleActivateModel(id: string): void {
    setModels(
      models.map((model) =>
        model.id === id ? { ...model, status: "active" } : model
      )
    );
  }

  function handleDeactivateModel(id: string): void {
    setModels(
      models.map((model) =>
        model.id === id ? { ...model, status: "inactive" } : model
      )
    );
  }

  return (
    <div className="container mx-auto p-4">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-3xl font-bold">Anomaly Detection Models</h1>
        <button className="rounded-md bg-primary px-4 py-2 text-primary-foreground">
          Add New Model
        </button>
      </div>

      <div className="mb-6 rounded-lg border p-4">
        <h2 className="mb-4 text-xl font-semibold">Model Performance</h2>
        <div className="grid gap-4 md:grid-cols-4">
          <div className="rounded-lg bg-background p-4 shadow-sm">
            <h3 className="mb-2 text-sm font-medium text-muted-foreground">
              Active Models
            </h3>
            <p className="text-2xl font-bold">
              {models.filter((m) => m.status === "active").length}
            </p>
          </div>
          
          <div className="rounded-lg bg-background p-4 shadow-sm">
            <h3 className="mb-2 text-sm font-medium text-muted-foreground">
              Average Accuracy
            </h3>
            <p className="text-2xl font-bold">
              {(
                models
                  .filter((m) => m.status === "active")
                  .reduce((acc, model) => acc + model.accuracy, 0) /
                models.filter((m) => m.status === "active").length
              ).toFixed(1)}%
            </p>
          </div>
          
          <div className="rounded-lg bg-background p-4 shadow-sm">
            <h3 className="mb-2 text-sm font-medium text-muted-foreground">
              Best Performing
            </h3>
            <p className="text-2xl font-bold">
              {
                models.reduce((best, model) => 
                  model.accuracy > best.accuracy ? model : best
                ).name.split(" ")[0]
              }
            </p>
          </div>
          
          <div className="rounded-lg bg-background p-4 shadow-sm">
            <h3 className="mb-2 text-sm font-medium text-muted-foreground">
              Models in Training
            </h3>
            <p className="text-2xl font-bold">
              {models.filter((m) => m.status === "training").length}
            </p>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="p-3 text-left font-medium">Name</th>
              <th className="p-3 text-left font-medium">Type</th>
              <th className="p-3 text-left font-medium">Status</th>
              <th className="p-3 text-left font-medium">Accuracy</th>
              <th className="p-3 text-left font-medium">Last Updated</th>
              <th className="p-3 text-left font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.id} className="border-b">
                <td className="p-3">
                  <div>
                    <p className="font-medium">{model.name}</p>
                    <p className="text-sm text-muted-foreground">{model.description}</p>
                  </div>
                </td>
                <td className="p-3">{model.type}</td>
                <td className="p-3">
                  <span className={`inline-block rounded px-2 py-1 text-xs font-medium ${getStatusBadgeClass(model.status)}`}>
                    {model.status.charAt(0).toUpperCase() + model.status.slice(1)}
                  </span>
                </td>
                <td className="p-3">
                  {model.status === "training" ? "-" : `${model.accuracy}%`}
                </td>
                <td className="p-3">{model.lastUpdated}</td>
                <td className="p-3">
                  <div className="flex space-x-2">
                    {model.status !== "active" && (
                      <button 
                        onClick={() => handleActivateModel(model.id)}
                        className="rounded bg-green-100 px-2 py-1 text-xs text-green-800"
                      >
                        Activate
                      </button>
                    )}
                    {model.status === "active" && (
                      <button 
                        onClick={() => handleDeactivateModel(model.id)}
                        className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-800"
                      >
                        Deactivate
                      </button>
                    )}
                    <button className="rounded bg-blue-100 px-2 py-1 text-xs text-blue-800">
                      Edit
                    </button>
                    <button className="rounded bg-red-100 px-2 py-1 text-xs text-red-800">
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Models;
