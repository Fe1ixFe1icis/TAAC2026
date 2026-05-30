import { useLenisInit } from '@/hooks/useLenis';
import ScrollVelocityWrapper from '@/components/effects/ScrollVelocityWrapper';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import HeroSection from '@/sections/HeroSection';
import OverviewSection from '@/sections/OverviewSection';
import ArchitectureSection from '@/sections/ArchitectureSection';
import CoreResultsSection from '@/sections/CoreResultsSection';
import OptimizationJourneySection from '@/sections/OptimizationJourneySection';
import KeyInsightsSection from '@/sections/KeyInsightsSection';
import FutureDirectionsSection from '@/sections/FutureDirectionsSection';
import ReferencesSection from '@/sections/ReferencesSection';

function App() {
  useLenisInit();

  return (
    <ScrollVelocityWrapper>
      <div className="relative bg-dark-bg min-h-screen">
        <Navigation />
        <main>
          <HeroSection />
          <OverviewSection />
          <ArchitectureSection />
          <CoreResultsSection />
          <OptimizationJourneySection />
          <KeyInsightsSection />
          <FutureDirectionsSection />
          <ReferencesSection />
        </main>
        <Footer />
      </div>
    </ScrollVelocityWrapper>
  );
}

export default App;
