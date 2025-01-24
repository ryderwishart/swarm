import { Outlet } from 'react-router-dom';

const RootLayout = () => {
  return (
    <div className="min-h-screen bg-background">
      <main className="container mx-auto py-8">
        <Outlet />
      </main>
    </div>
  );
};

export default RootLayout;
