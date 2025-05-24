import React from 'react';
import Navbar from './Navbar';

interface PageLayoutProps {
  children: React.ReactNode;
}

const PageLayout: React.FC<PageLayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />
      <main className="flex-1 container mx-auto px-4 py-8">
        {children}
      </main>
      <footer className="bg-card py-6 border-t border-tertiary">
        <div className="container mx-auto px-4 text-center text-sm">
          &copy; {new Date().getFullYear()} <span className="text-primary font-medium">DATA<span className="text-secondary">TIKA</span></span>. All rights reserved.
        </div>
      </footer>
    </div>
  );
};

export default PageLayout;