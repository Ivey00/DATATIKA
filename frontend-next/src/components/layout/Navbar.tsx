"use client";

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, BarChart, ActivitySquare, Clock, Image } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useIsMobile } from '@/hooks/use-mobile';

const Navbar = () => {
  const isMobile = useIsMobile();
  const pathname = usePathname();
  
  const navItems = [
    { 
      name: 'Dashboard', 
      path: '/', 
      icon: <LayoutDashboard className="h-5 w-5" /> 
    },
    { 
      name: 'Target Prediction', 
      path: '/target-prediction', 
      icon: <BarChart className="h-5 w-5" /> 
    },
    { 
      name: 'Anomaly Detection', 
      path: '/anomaly-detection', 
      icon: <ActivitySquare className="h-5 w-5" /> 
    },
    { 
      name: 'Numerical Classifier', 
      path: '/numerical-classifier', 
      icon: <BarChart className="h-5 w-5 rotate-90" /> 
    },
    { 
      name: 'Time Series', 
      path: '/time-series', 
      icon: <Clock className="h-5 w-5" /> 
    },
    { 
      name: 'Image Classification', 
      path: '/image-classification', 
      icon: <Image className="h-5 w-5" /> 
    }
  ];

  return (
    <nav className="bg-card shadow-sm border-b border-tertiary">
      <div className="container mx-auto px-4 py-3">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <h1 className="text-2xl font-bold text-primary hover:text-secondary transition-colors cursor-pointer">DATATIKA</h1>
          </div>
          
          {!isMobile && (
            <div className="flex space-x-1">
              {navItems.map((item) => {
                const isActive = pathname === item.path;
                return (
                  <Link 
                    key={item.path} 
                    href={item.path}
                    className={`px-3 py-2 rounded-md flex items-center gap-1.5 text-sm font-medium transition-colors
                     ${isActive 
                      ? 'bg-primary text-primary-foreground' 
                      : 'text-foreground hover:bg-secondary hover:text-secondary-foreground'}`
                    }
                  >
                    {item.icon}
                    {item.name}
                  </Link>
                );
              })}
            </div>
          )}
          
          <div>
            <Button variant="secondary" size="sm">
              Sign In
            </Button>
          </div>
        </div>
      </div>
      
      {isMobile && (
        <div className="overflow-x-auto pb-2 px-4">
          <div className="flex space-x-1 w-max">
            {navItems.map((item) => {
              const isActive = pathname === item.path;
              return (
                <Link 
                  key={item.path} 
                  href={item.path}
                  className={`px-3 py-2 rounded-md flex items-center gap-1.5 text-sm font-medium whitespace-nowrap transition-colors
                   ${isActive 
                    ? 'bg-primary text-primary-foreground' 
                    : 'text-foreground hover:bg-secondary hover:text-secondary-foreground'}`
                  }
                >
                  {item.icon}
                  {item.name}
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;