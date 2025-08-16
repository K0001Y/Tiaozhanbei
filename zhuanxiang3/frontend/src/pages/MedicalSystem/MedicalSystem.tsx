import React from 'react';
import {
  DashboardModule,
  KnowledgeModule,
  SearchModule,
  DiagnosisModule,
  RecordModule,
  InquiryModule,
  AIAssistModule,
} from './modules';
import './MedicalSystem.scss';

interface MedicalSystemProps {
  activeModule?: string;
}

const MedicalSystem: React.FC<MedicalSystemProps> = ({ activeModule = 'dashboard' }) => {
  const renderModule = () => {
    switch (activeModule) {
      case 'dashboard':
        return <DashboardModule />;
      case 'knowledge':
        return <KnowledgeModule />;
      case 'search':
        return <SearchModule />;
      case 'diagnosis':
        return <DiagnosisModule />;
      case 'record':
        return <RecordModule />;
      case 'inquiry':
        return <InquiryModule />;
      case 'ai-assist':
        return <AIAssistModule />;
      default:
        return <DashboardModule />;
    }
  };

  return (
    <div className="medical-system">
      {renderModule()}
    </div>
  );
};

export default MedicalSystem;
