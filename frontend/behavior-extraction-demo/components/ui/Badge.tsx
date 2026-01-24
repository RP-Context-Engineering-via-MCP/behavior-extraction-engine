import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  className?: string;
  colorClass?: string;
}

export const Badge: React.FC<BadgeProps> = ({ children, className = '', colorClass = 'bg-gray-100 text-gray-800' }) => {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colorClass} ${className}`}>
      {children}
    </span>
  );
};
