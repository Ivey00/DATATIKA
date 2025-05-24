import Link from 'next/link';
import { Button } from '@/components/ui/button';
import PageLayout from '@/components/layout/PageLayout';

export default function NotFound() {
  return (
    <PageLayout>
      <div className="flex flex-col items-center justify-center py-20">
        <h1 className="text-6xl font-bold text-primary">404</h1>
        <h2 className="text-2xl font-medium mb-6 mt-2">Page Not Found</h2>
        <p className="text-muted-foreground mb-8 text-center max-w-md">
          The page you're looking for doesn't exist or is still under development.
        </p>
        <Link href="/">
          <Button>
            Return to Dashboard
          </Button>
        </Link>
      </div>
    </PageLayout>
  );
}