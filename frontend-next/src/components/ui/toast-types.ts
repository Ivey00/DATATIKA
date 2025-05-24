import * as React from "react"

// Define the ToastProps type to match what's expected by both toast.tsx and use-toast.ts
export interface ToastProps {
  id?: string
  className?: string
  variant?: "default" | "destructive"
  open?: boolean
  onOpenChange?: (open: boolean) => void
  title?: React.ReactNode
  description?: React.ReactNode
  action?: ToastActionElement
  // Add other HTML attributes that might be needed
  style?: React.CSSProperties
  role?: string
  "aria-live"?: "assertive" | "off" | "polite"
}

// Define the ToastActionElement type
export type ToastActionElement = React.ReactElement<{
  className?: string
  altText?: string
  onClick?: () => void
}>