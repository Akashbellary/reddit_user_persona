import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import { ErrorBoundary } from "@/components/error-boundary"
import { ToastProvider } from "@/components/toast-provider"
import { Suspense } from "react"
import "./globals.css"

export const metadata: Metadata = {
  title: "Reddit Persona Analyzer - AI-Powered User Psychology Analysis",
  description:
    "Analyze Reddit profiles to generate detailed psychological personas using advanced AI. Understand personality traits, communication styles, and behavioral patterns.",
  generator: "v0.app",
  keywords: ["reddit", "psychology", "ai", "persona", "analysis", "personality"],
  authors: [{ name: "Reddit Persona Analyzer" }],
  openGraph: {
    title: "Reddit Persona Analyzer",
    description: "AI-powered psychological analysis of Reddit users",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Reddit Persona Analyzer",
    description: "AI-powered psychological analysis of Reddit users",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable}`}>
        <Suspense fallback={<div>Loading...</div>}>
          <ErrorBoundary>
            {children}
            <ToastProvider />
          </ErrorBoundary>
        </Suspense>
        <Analytics />
      </body>
    </html>
  )
}
