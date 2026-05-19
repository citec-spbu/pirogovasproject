import { Outlet } from 'react-router-dom';
import { AdminHeader } from '../../../widgets/AdminHeader/AdminHeader';
import { AdminSidebar } from '../../../widgets/AdminSidebar/AdminSidebar';
import cls from './AdminLayout.module.scss';

export const AdminLayout = () => {
  return (
    <div className={cls.layout}>
      <AdminHeader />

      <div className={cls.body}>
        <AdminSidebar />

        <main className={cls.content}>
          <Outlet />
        </main>
      </div>
    </div>
  );
};