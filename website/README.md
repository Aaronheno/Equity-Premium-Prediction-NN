# Neural Networks Documentation Website

A comprehensive, interactive documentation website for neural network architectures in equity premium prediction, built with Next.js, TypeScript, and Tailwind CSS.

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn package manager

### Installation

1. **Navigate to the website directory:**
   ```bash
   cd website
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 📁 Project Structure

```
website/
├── app/                    # Next.js 13+ app directory
│   ├── globals.css        # Global styles and CSS variables
│   ├── layout.tsx         # Root layout component
│   └── page.tsx           # Homepage
├── components/            # Reusable React components
│   ├── Navigation.tsx     # Main navigation bar
│   ├── Hero.tsx          # Landing page hero section
│   ├── OverviewSection.tsx
│   ├── ModelArchitectures.tsx
│   ├── KeyFeatures.tsx
│   └── Footer.tsx
├── public/               # Static assets
├── next.config.js        # Next.js configuration
├── tailwind.config.js    # Tailwind CSS configuration
├── tsconfig.json         # TypeScript configuration
└── package.json          # Dependencies and scripts
```

## 🎨 Design System

### Color Palette
- **Primary Background:** `#0a0b0f`
- **Secondary Background:** `#1a1b23`
- **Tertiary Background:** `#2a2d3a`
- **Accent Blue:** `#3b82f6`
- **Accent Purple:** `#8b5cf6`
- **Accent Green:** `#10b981`
- **Accent Orange:** `#f59e0b`

### Typography
- **Sans-serif:** Inter
- **Monospace:** JetBrains Mono
- **Math:** Computer Modern

### Key Features
- Dark professional theme
- Responsive design (mobile-first)
- Smooth animations with Framer Motion
- Syntax highlighting for code blocks
- Mathematical formula rendering with KaTeX
- Interactive components and visualizations

## 🛠 Technology Stack

- **Framework:** Next.js 14 with App Router
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Animations:** Framer Motion
- **Code Highlighting:** React Syntax Highlighter
- **Math Rendering:** KaTeX & React-KaTeX
- **Icons:** Lucide React
- **Visualizations:** D3.js (planned)

## 📝 Content Integration

The website integrates content from the main project:

### Source Files Referenced:
- `../NEURAL_NETWORKS_EXPLAINED.md` - Main documentation
- `../src/models/nns.py` - Neural network architectures
- `../src/configs/search_spaces.py` - Hyperparameter configurations
- `../src/experiments/` - Experiment scripts
- `../data/ml_equity_premium_data.xlsx` - Dataset
- `../runs/` - Experiment results

### Planned Pages:
1. **Introduction** - Project overview and getting started
2. **Data & Setup** - 31 financial indicators and problem formulation
3. **Data Preprocessing** - Scaling, temporal splits, feature engineering
4. **Model Architecture** - Net1-Net5 and DNet1-DNet3 detailed explanation
5. **Forward Pass** - Information flow through neural networks
6. **Loss Calculation** - MSE, L1, and L2 regularization
7. **Backpropagation** - Gradient computation and chain rule
8. **Optimization** - Adam, RMSprop, SGD optimizers
9. **Hyperparameter Optimization** - Bayesian, grid, and random search
10. **Making Predictions** - Model inference and output processing
11. **Evaluation** - Performance metrics and statistical tests
12. **Complete Pipeline** - End-to-end implementation
13. **Interactive Architecture** - Visualize all 8 model architectures

## 🎯 Planned Features

### Interactive Components
- [ ] Architecture visualization tool
- [ ] Hyperparameter space exploration
- [ ] Training progress animations
- [ ] Performance comparison charts
- [ ] Code execution playground
- [ ] Mathematical formula explorer

### Content Features
- [ ] Copy-to-clipboard for code blocks
- [ ] Progressive disclosure of complex topics
- [ ] Search functionality across all content
- [ ] Bookmarking and progress tracking
- [ ] Dark/light theme toggle
- [ ] Print-friendly layouts

### Performance Features
- [ ] Static site generation (SSG)
- [ ] Image optimization
- [ ] Code splitting and lazy loading
- [ ] Service worker for offline access
- [ ] Progressive web app (PWA) capabilities

## 🚢 Deployment

### Build for Production
```bash
npm run build
npm start
```

### Deployment Options
- **Vercel** (recommended for Next.js)
- **Netlify**
- **GitHub Pages** (with static export)

### Environment Variables
Create a `.env.local` file for any required environment variables:
```
# Add any API keys or configuration here
NEXT_PUBLIC_SITE_URL=https://your-domain.com
```

## 📱 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🤝 Contributing

This website is built to showcase the neural networks documentation. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Neural Networks for Equity Premium Prediction documentation suite.

## 🙏 Acknowledgments

- Built with Next.js and the React ecosystem
- Mathematical rendering powered by KaTeX
- Animations created with Framer Motion
- Icons provided by Lucide React
- Styling with Tailwind CSS

---

For more information about the neural networks implementation, see the main `NEURAL_NETWORKS_EXPLAINED.md` documentation.