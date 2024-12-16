import { Outlet } from 'react-router-dom';

const RootLayout = () => {
  return (
    <div className="min-h-screen bg-background max-w-lg mx-auto">
      <main className="container mx-auto py-8 px-4">
        <Outlet />
      </main>
    </div>
  );
};

export default RootLayout;
