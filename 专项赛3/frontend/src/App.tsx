import React, {useState} from 'react'
import './App.scss'
import AICopilot from './pages/AICopilot/AICopilot'
import Hear from './pages/Hear/Hear'
import Library from './pages/Library/Library'
import Search from './pages/Search/Search'
import User from './pages/User/User'
import Watch from './pages/Watch/Watch'
import Login from './pages/Auth/Login/Login'
import Register from './pages/Auth/Register/Register'
import Navbar from './components/Navbar/Navbar'
import Tabbar from './components/Tabbar/Tabbar'
import {
  createBrowserRouter,
  RouterProvider,
  // Navigate,
} from "react-router-dom";

function App() {

  const [currentSection, setCurrentSection] = useState<string>("AI辅助")

  const router = createBrowserRouter([
    // {
    //   path: "*",
    //   element: isAuthenticated ? (
    //     <Navigate to="/AICopilot" replace />
    //   ) : (
    //     <Navigate to="/auth/login" replace />
    //   ),
    // },

    {
      path: "/AICopilot",
      element: <AICopilot />,
    },

    {
      path: "/auth/login",
      element: <Login />,
    },

    {
      path: "/auth/register",
      element: <Register />,
    },

    {
      path: "/Hear",
      element: <Hear />,
    },

    {
      path: "/Library",
      element: <Library />,
    },

    {
      path: "/Search",
      element: <Search />,
    },

    {
      path: "/User",
      element: <User />,
    },

    {
      path: "/Watch",
      element: <Watch />,
    }
    
  ])
  
  return (
    <div className="App">
      <React.Suspense fallback={<div>加载中......请稍候......</div>}>
        <Navbar section={currentSection} username='用户名'/>
        <RouterProvider router={router} />
        <Tabbar setCurrentSection={setCurrentSection} />
      </React.Suspense>
    </div>
  )
}

export default App
