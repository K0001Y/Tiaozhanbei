import React from 'react'
import './App.scss'
import { 
  Layout, 
  ProtectedRoute, 
  PublicRoute 
} from './components'
import { 
  MedicalSystem, 
  Login, 
  Register, 
  NotFound 
} from './pages'
import {
  createBrowserRouter,
  RouterProvider,
  Navigate,
} from "react-router-dom";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Navigate to="/dashboard" replace />,
  },
  {
    path: "/dashboard",
    element: (
      //<ProtectedRoute>
        <Layout>
          <MedicalSystem />
        </Layout>
      //</ProtectedRoute>
    ),
  },
  {
    path: "/medical",
    element: (
      //<ProtectedRoute>
        <Layout>
          <MedicalSystem />
        </Layout>
      //</ProtectedRoute>
    ),
  },
  {
    path: "/login",
    element: (
      <PublicRoute>
        <Login />
      </PublicRoute>
    ),
  },
  {
    path: "/register", 
    element: (
      <PublicRoute>
        <Register />
      </PublicRoute>
    ),
  },
  {
    path: "*",
    element: <NotFound />,
  },
])

function App() {
  return (
    <div className="App">
      <React.Suspense fallback={<div className="loading">加载中......请稍候......</div>}>
        <RouterProvider router={router} />
      </React.Suspense>
    </div>
  )
}

export default App
