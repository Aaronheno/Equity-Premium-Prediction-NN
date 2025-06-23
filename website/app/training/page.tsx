// This file has been replaced by individual training pages:
// - /forward-pass
// - /loss-calculation 
// - /backpropagation
// - /optimization
//
// Please navigate to the specific training topics using the navigation menu.

export default function TrainingPage() {
  return (
    <div className="min-h-screen bg-bg-primary pt-20 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-text-primary mb-4">Training Topics</h1>
        <p className="text-text-secondary mb-6">
          Training content has been split into focused sections:
        </p>
        <div className="space-y-2">
          <div>• Forward Pass</div>
          <div>• Loss Calculation</div>
          <div>• Backpropagation</div>
          <div>• Optimization</div>
        </div>
        <p className="text-text-muted mt-6">Use the navigation menu to access specific topics.</p>
      </div>
    </div>
  )
}