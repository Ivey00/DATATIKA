import ModelDetailsContent from './ModelDetailsContent';

export default function ModelDetailsPage({ params }: { params: { id: string } }) {
  return <ModelDetailsContent modelId={params.id} />;
} 