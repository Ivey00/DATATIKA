import PredictContent from './PredictContent';

export default async function PredictPage({ params }: { params: { id: string } }) {
  const modelId = await Promise.resolve(params.id);
  return <PredictContent modelId={modelId} />;
} 