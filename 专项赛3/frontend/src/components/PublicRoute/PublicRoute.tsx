import React from 'react';
import { Navigate } from 'react-router-dom';
import { useUserStore } from '../../store/userStore';

interface PublicRouteProps {
  children: React.ReactNode;
}

const PublicRoute: React.FC<PublicRouteProps> = ({ children }) => {
  const { isAuthenticated } = useUserStore();

  if (isAuthenticated) {
    // 如果已经登录，重定向到主页
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

export default PublicRoute;
