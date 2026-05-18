import cls from './AdminHeader.module.scss';

export const AdminHeader = () => {
  return (
    <header className={cls.header}>
      <span className={cls.logo}>CTReport Admin</span>
    </header>
  );
};