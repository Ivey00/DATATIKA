"use client";

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { LayoutDashboard, BarChart, ActivitySquare, Clock, Image, LogOut, User, Database } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useIsMobile } from '@/hooks/use-mobile';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface User {
  id: number;
  name: string;
  email: string;
}

const Navbar = () => {
  const isMobile = useIsMobile();
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/auth/me', {
          credentials: 'include',
        });
        if (response.ok) {
          const data = await response.json();
          setUser(data.user);
        }
      } catch (error) {
        console.error('Auth check failed:', error);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  const handleSignOut = async () => {
    try {
      await fetch('http://localhost:8000/api/auth/signout', {
        method: 'POST',
        credentials: 'include',
      });
      setUser(null);
      router.push('/');
    } catch (error) {
      console.error('Sign out failed:', error);
    }
  };

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
    },
    {
      name: 'My Models',
      path: '/my-models',
      icon: <Database className="h-5 w-5" />
    }
  ];

  return (
    <nav className="bg-card shadow-sm border-b border-tertiary">
      <div className="container mx-auto px-4 py-3">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <Link href="/">
              <h1 className="text-2xl font-bold text-primary hover:text-secondary transition-colors cursor-pointer">
                DATATIKA
              </h1>
            </Link>
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
          
          <div className="flex items-center gap-4">
            {!isLoading && (
              user ? (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button 
                      variant="ghost" 
                      className="flex items-center gap-2 hover:bg-secondary"
                    >
                      <User className="h-5 w-5" />
                      <span className="font-medium">{user.name}</span>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-48">
                    <DropdownMenuItem 
                      className="text-destructive focus:text-destructive cursor-pointer"
                      onClick={handleSignOut}
                    >
                      <LogOut className="h-4 w-4 mr-2" />
                      Sign Out
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              ) : (
                <Link href="/signin">
                  <Button variant="secondary" size="sm">
                    Sign In
                  </Button>
                </Link>
              )
            )}
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